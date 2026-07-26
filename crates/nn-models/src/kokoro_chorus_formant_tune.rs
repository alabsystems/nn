// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vocal formant resonance tuner for Kokoro chorus voices.
//!
//! Shifts, enhances, or blends formant frequencies across chorus voices to
//! create subtle timbre differences (e.g., nasal, chesty, bright, dark)
//! without changing pitch. This enables realistic ensemble variation where
//! each voice has a slightly different vocal tract character.
//!
//! # Algorithm
//!
//! 1. **LPC analysis** via Levinson-Durbin recursion on windowed frames
//!    to estimate the vocal tract transfer function.
//! 2. **Formant detection** by peak-picking the LPC magnitude response
//!    to find F1-F4 center frequencies.
//! 3. **Per-formant peaking EQ** filters (second-order biquad) to shift,
//!    boost, or cut each detected formant independently.
//! 4. **Dry/wet blend** to control the amount of formant modification.
//!
//! # References
//!
//! - Makhoul, J. "Linear Prediction: A Tutorial Review." Proceedings of
//!   the IEEE, 63(4), 1975.
//! - Markel, J. D. & Gray, A. H. "Linear Prediction of Speech."
//!   Springer-Verlag, 1976.
//! - Smith, J. O. "Introduction to Digital Filters with Audio Applications."
//!   <https://ccrma.stanford.edu/~jos/filters/>
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default LPC order for formant analysis. Order 10-12 is standard for
/// speech at 8-24 kHz; we use 12 to capture F1-F4 reliably at 24 kHz.
const DEFAULT_LPC_ORDER: usize = 12;

/// Number of bins to evaluate in the LPC magnitude response for peak picking.
const LPC_SPECTRUM_BINS: usize = 512;

/// Minimum formant frequency we consider valid (Hz).
const MIN_FORMANT_HZ: f32 = 200.0;

/// Maximum formant frequency we consider valid (Hz).
const MAX_FORMANT_HZ: f32 = 5500.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-formant tuning parameters: frequency offset, bandwidth scale, and gain.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct FormantBand {
    /// Frequency offset in Hz applied to the detected formant center.
    /// Positive shifts the formant up, negative shifts it down.
    /// Range: -500.0 to 500.0. Default: 0.0.
    pub shift_hz: f32,

    /// Bandwidth scale factor. 1.0 = natural bandwidth, <1.0 narrows
    /// (more resonant), >1.0 widens (more diffuse).
    /// Range: 0.25 to 4.0. Default: 1.0.
    pub bandwidth_scale: f32,

    /// Gain in dB applied to this formant. Positive boosts, negative cuts.
    /// Range: -12.0 to 12.0. Default: 0.0.
    pub gain_db: f32,
}

impl Default for FormantBand {
    fn default() -> Self {
        Self {
            shift_hz: 0.0,
            bandwidth_scale: 1.0,
            gain_db: 0.0,
        }
    }
}

impl FormantBand {
    /// Create a new formant band with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the frequency shift in Hz.
    #[must_use]
    pub fn with_shift_hz(mut self, hz: f32) -> Self {
        self.shift_hz = hz;
        self
    }

    /// Set the bandwidth scale factor.
    #[must_use]
    pub fn with_bandwidth_scale(mut self, scale: f32) -> Self {
        self.bandwidth_scale = scale;
        self
    }

    /// Set the gain in dB.
    #[must_use]
    pub fn with_gain_db(mut self, db: f32) -> Self {
        self.gain_db = db;
        self
    }
}

/// Configuration for the vocal formant resonance tuner.
///
/// Constructed via [`FormantTuneConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FormantTuneConfig {
    /// Number of formants to detect and process (1-4). Default: 4.
    pub n_formants: usize,

    /// Per-formant tuning for each voice. Outer index = voice, inner = formant.
    /// If a voice index is absent, neutral (identity) tuning is used.
    pub voice_formants: Vec<Vec<FormantBand>>,

    /// Global blend amount: 0.0 = fully dry (no effect), 1.0 = fully wet.
    /// Default: 1.0.
    pub blend_amount: f32,

    /// LPC analysis order. Higher = finer spectral detail, more CPU.
    /// Range: 6-20. Default: 12.
    pub lpc_order: usize,

    /// Analysis frame size in samples. Must be >= 64. Default: 1024.
    pub frame_size: usize,

    /// Hop size between analysis frames. Must be in [1, frame_size].
    /// Default: 512.
    pub hop_size: usize,

    /// Sample rate in Hz. Default: 24000.0 (Kokoro native rate).
    pub sample_rate: f32,
}

impl Default for FormantTuneConfig {
    fn default() -> Self {
        Self {
            n_formants: 4,
            voice_formants: Vec::new(),
            blend_amount: 1.0,
            lpc_order: DEFAULT_LPC_ORDER,
            frame_size: 1024,
            hop_size: 512,
            sample_rate: KOKORO_SAMPLE_RATE as f32,
        }
    }
}

impl FormantTuneConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of formants to detect.
    #[must_use]
    pub fn with_n_formants(mut self, n: usize) -> Self {
        self.n_formants = n;
        self
    }

    /// Set the per-voice formant tuning.
    #[must_use]
    pub fn with_voice_formants(mut self, v: Vec<Vec<FormantBand>>) -> Self {
        self.voice_formants = v;
        self
    }

    /// Set the global blend amount.
    #[must_use]
    pub fn with_blend_amount(mut self, blend: f32) -> Self {
        self.blend_amount = blend;
        self
    }

    /// Set the LPC analysis order.
    #[must_use]
    pub fn with_lpc_order(mut self, order: usize) -> Self {
        self.lpc_order = order;
        self
    }

    /// Set the analysis frame size.
    #[must_use]
    pub fn with_frame_size(mut self, size: usize) -> Self {
        self.frame_size = size;
        self
    }

    /// Set the hop size.
    #[must_use]
    pub fn with_hop_size(mut self, hop: usize) -> Self {
        self.hop_size = hop;
        self
    }

    /// Set the sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, sr: f32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.n_formants == 0 || self.n_formants > 4 {
            return Err(KokoroError::InvalidConfig {
                field: "n_formants",
                reason: format!("must be in [1, 4], got {}", self.n_formants),
            });
        }
        if !self.blend_amount.is_finite() || !(0.0..=1.0).contains(&self.blend_amount) {
            return Err(KokoroError::InvalidConfig {
                field: "blend_amount",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.blend_amount
                ),
            });
        }
        if self.lpc_order < 6 || self.lpc_order > 20 {
            return Err(KokoroError::InvalidConfig {
                field: "lpc_order",
                reason: format!("must be in [6, 20], got {}", self.lpc_order),
            });
        }
        if self.frame_size < 64 {
            return Err(KokoroError::InvalidConfig {
                field: "frame_size",
                reason: format!("must be >= 64, got {}", self.frame_size),
            });
        }
        if self.hop_size == 0 || self.hop_size > self.frame_size {
            return Err(KokoroError::InvalidConfig {
                field: "hop_size",
                reason: format!(
                    "must be in [1, frame_size={}], got {}",
                    self.frame_size, self.hop_size
                ),
            });
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be finite and positive, got {}", self.sample_rate),
            });
        }
        // Validate per-formant bands.
        for (vi, bands) in self.voice_formants.iter().enumerate() {
            for (fi, band) in bands.iter().enumerate() {
                if !band.shift_hz.is_finite() || band.shift_hz < -500.0 || band.shift_hz > 500.0 {
                    return Err(KokoroError::InvalidConfig {
                        field: "shift_hz",
                        reason: format!(
                            "voice[{vi}].formant[{fi}].shift_hz = {}: must be in [-500, 500]",
                            band.shift_hz
                        ),
                    });
                }
                if !band.bandwidth_scale.is_finite()
                    || band.bandwidth_scale < 0.25
                    || band.bandwidth_scale > 4.0
                {
                    return Err(KokoroError::InvalidConfig {
                        field: "bandwidth_scale",
                        reason: format!(
                            "voice[{vi}].formant[{fi}].bandwidth_scale = {}: must be in [0.25, 4.0]",
                            band.bandwidth_scale
                        ),
                    });
                }
                if !band.gain_db.is_finite() || band.gain_db < -12.0 || band.gain_db > 12.0 {
                    return Err(KokoroError::InvalidConfig {
                        field: "gain_db",
                        reason: format!(
                            "voice[{vi}].formant[{fi}].gain_db = {}: must be in [-12, 12]",
                            band.gain_db
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    // -- Presets ---------------------------------------------------------------

    /// Natural preset: no formant modification (identity).
    #[must_use]
    pub fn natural() -> Self {
        Self::default().with_blend_amount(0.0)
    }

    /// Bright preset: shift F3 up by 200 Hz for added presence.
    #[must_use]
    pub fn bright() -> Self {
        let bands = vec![
            FormantBand::new(),
            FormantBand::new(),
            FormantBand::new().with_shift_hz(200.0).with_gain_db(2.0),
            FormantBand::new(),
        ];
        Self::default().with_voice_formants(vec![bands])
    }

    /// Dark preset: shift F2 down by 100 Hz for a warmer character.
    #[must_use]
    pub fn dark() -> Self {
        let bands = vec![
            FormantBand::new(),
            FormantBand::new().with_shift_hz(-100.0).with_gain_db(-1.0),
            FormantBand::new(),
            FormantBand::new(),
        ];
        Self::default().with_voice_formants(vec![bands])
    }

    /// Nasal preset: boost F1 and F3 for a nasal quality.
    #[must_use]
    pub fn nasal() -> Self {
        let bands = vec![
            FormantBand::new()
                .with_gain_db(4.0)
                .with_bandwidth_scale(0.7),
            FormantBand::new(),
            FormantBand::new()
                .with_gain_db(3.0)
                .with_bandwidth_scale(0.8),
            FormantBand::new(),
        ];
        Self::default().with_voice_formants(vec![bands])
    }

    /// Chest preset: boost F1, cut F3 for a deeper chesty quality.
    #[must_use]
    pub fn chest() -> Self {
        let bands = vec![
            FormantBand::new().with_gain_db(4.0).with_shift_hz(-30.0),
            FormantBand::new(),
            FormantBand::new().with_gain_db(-3.0),
            FormantBand::new(),
        ];
        Self::default().with_voice_formants(vec![bands])
    }
}

// ---------------------------------------------------------------------------
// Formant Tuner
// ---------------------------------------------------------------------------

/// Stateful vocal formant resonance tuner.
///
/// Analyzes each voice's formant structure via LPC and applies per-formant
/// peaking EQ filters to shift, boost, or cut formant frequencies.
pub struct FormantTuner {
    config: FormantTuneConfig,
    /// Pre-computed Hann window for analysis frames.
    window: Vec<f32>,
}

impl FormantTuner {
    /// Create a new formant tuner from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is invalid.
    pub fn new(config: FormantTuneConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let window = hann_window(config.frame_size);
        Ok(Self { config, window })
    }

    /// Create a formant tuner with default configuration.
    pub fn with_defaults() -> Result<Self, KokoroError> {
        Self::new(FormantTuneConfig::default())
    }

    /// Reset the tuner (stateless analysis, so this is a no-op placeholder
    /// for API consistency with other chorus processors).
    pub fn reset(&mut self) {
        // LPC analysis is frame-by-frame with no inter-frame state.
        // Biquad filters are created per-process call. Nothing to reset.
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &FormantTuneConfig {
        &self.config
    }

    /// Detect formant frequencies (F1-F4) from an audio signal via LPC.
    ///
    /// Returns a vector of detected formant center frequencies in Hz,
    /// sorted ascending. The number of entries is at most `n_formants`.
    #[must_use]
    pub fn detect_formants(&self, audio: &[f32]) -> Vec<f32> {
        if audio.len() < self.config.frame_size {
            return Vec::new();
        }

        // Analyze a single frame from the middle of the signal for
        // a representative formant snapshot.
        let mid = audio.len().saturating_sub(self.config.frame_size) / 2;
        let frame = &audio[mid..mid + self.config.frame_size];

        // Apply Hann window.
        let windowed: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| if s.is_finite() { s * w } else { 0.0 })
            .collect();

        // LPC via Levinson-Durbin.
        let lpc_coeffs = levinson_durbin(&windowed, self.config.lpc_order);

        // Peak-pick the LPC magnitude response to find formants.
        

        lpc_peak_pick(&lpc_coeffs, self.config.sample_rate, self.config.n_formants)
    }

    /// Process a single voice buffer in-place.
    ///
    /// Detects formants, then applies per-formant peaking EQ based on the
    /// voice's formant band configuration.
    pub fn process_voice(&mut self, audio: &mut [f32], voice_index: usize) {
        // Bypass if blend is effectively zero.
        if self.config.blend_amount < 1e-6 {
            return;
        }

        // Get the formant bands for this voice (or use neutral defaults).
        let bands: Vec<FormantBand> = self
            .config
            .voice_formants
            .get(voice_index)
            .cloned()
            .unwrap_or_default();

        // If no bands configured, nothing to do.
        if bands.is_empty() {
            return;
        }

        // Detect formant frequencies from the current audio.
        let formants = self.detect_formants(audio);
        if formants.is_empty() {
            return;
        }

        // Save dry copy for blending.
        let dry: Vec<f32> = audio.to_vec();

        // Apply per-formant peaking EQ.
        let n_apply = formants.len().min(bands.len());
        for i in 0..n_apply {
            let band = &bands[i];

            // Skip bands with no modification.
            if band.shift_hz.abs() < 1e-3
                && (band.gain_db).abs() < 1e-3
                && (band.bandwidth_scale - 1.0).abs() < 1e-3
            {
                continue;
            }

            let center_hz = formants[i] + band.shift_hz;
            // Clamp to valid range.
            let center_hz = center_hz.clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ);

            // Approximate formant bandwidth: wider formants have bandwidth
            // proportional to their center frequency (~10-15% of center).
            let base_bw = center_hz * 0.12;
            let bw = base_bw * band.bandwidth_scale;

            apply_peaking_eq(audio, center_hz, bw, band.gain_db, self.config.sample_rate);
        }

        // Blend wet with dry.
        let blend = self.config.blend_amount;
        if blend < 1.0 - 1e-6 {
            let dry_mix = 1.0 - blend;
            for (wet, &d) in audio.iter_mut().zip(dry.iter()) {
                *wet = *wet * blend + d * dry_mix;
                if !wet.is_finite() {
                    *wet = 0.0;
                }
            }
        } else {
            // Full wet: just ensure finite output.
            for s in audio.iter_mut() {
                if !s.is_finite() {
                    *s = 0.0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: Hann window
// ---------------------------------------------------------------------------

fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let scale = 2.0 * std::f64::consts::PI / (n - 1) as f64;
    (0..n)
        .map(|i| (0.5 * (1.0 - (scale * i as f64).cos())) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Internal: Levinson-Durbin LPC
// ---------------------------------------------------------------------------

/// Compute LPC coefficients via Levinson-Durbin recursion.
///
/// Returns `order + 1` coefficients where `a[0] = 1.0`.
fn levinson_durbin(signal: &[f32], order: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || order == 0 {
        return vec![1.0];
    }

    // Compute autocorrelation R[0..order].
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0f64;
        for i in lag..n {
            let s0 = f64::from(signal[i]);
            let s1 = f64::from(signal[i - lag]);
            if s0.is_finite() && s1.is_finite() {
                sum += s0 * s1;
            }
        }
        r[lag] = sum;
    }

    // Trivial or silent signal.
    if r[0].abs() < 1e-30 {
        let mut a = vec![0.0f64; order + 1];
        a[0] = 1.0;
        return a;
    }

    // Levinson-Durbin recursion.
    let mut a = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut err = r[0];

    for m in 1..=order {
        // Compute reflection coefficient.
        let mut lambda = 0.0f64;
        for j in 1..m {
            lambda += a[j] * r[m - j];
        }
        lambda = -(r[m] + lambda) / err;

        // Guard against instability.
        if !lambda.is_finite() || lambda.abs() >= 1.0 {
            break;
        }

        // Update coefficients.
        let mut a_new = a.clone();
        a_new[m] = lambda;
        for j in 1..m {
            a_new[j] = a[j] + lambda * a[m - j];
        }
        a = a_new;

        err *= 1.0 - lambda * lambda;
        if err < 1e-30 {
            break;
        }
    }

    a
}

// ---------------------------------------------------------------------------
// Internal: LPC magnitude response peak picking
// ---------------------------------------------------------------------------

/// Evaluate the LPC magnitude response and find formant peaks.
///
/// Returns up to `n_formants` peak frequencies in Hz, sorted ascending.
fn lpc_peak_pick(lpc_coeffs: &[f64], sample_rate: f32, n_formants: usize) -> Vec<f32> {
    let n_bins = LPC_SPECTRUM_BINS;
    let sr = f64::from(sample_rate);

    // Evaluate |1/A(e^jw)| at uniformly spaced frequencies.
    let mut magnitudes = Vec::with_capacity(n_bins);
    for k in 0..n_bins {
        let freq = (k as f64 / n_bins as f64) * (sr / 2.0);
        let omega = 2.0 * std::f64::consts::PI * freq / sr;

        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, &coeff) in lpc_coeffs.iter().enumerate() {
            re += coeff * (omega * i as f64).cos();
            im -= coeff * (omega * i as f64).sin();
        }

        let mag_sq = re * re + im * im;
        let inv_mag = if mag_sq > 1e-30 {
            1.0 / mag_sq.sqrt()
        } else {
            0.0
        };
        magnitudes.push(inv_mag);
    }

    // Find local maxima (peaks) in the magnitude response.
    let mut peaks: Vec<(f32, f64)> = Vec::new(); // (freq_hz, magnitude)
    for k in 1..n_bins.saturating_sub(1) {
        if magnitudes[k] > magnitudes[k - 1] && magnitudes[k] > magnitudes[k + 1] {
            let freq_hz = (k as f64 / n_bins as f64 * (sr / 2.0)) as f32;
            if (MIN_FORMANT_HZ..=MAX_FORMANT_HZ).contains(&freq_hz) {
                peaks.push((freq_hz, magnitudes[k]));
            }
        }
    }

    // Sort by magnitude descending to pick the strongest peaks.
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take the top n_formants peaks.
    let mut formants: Vec<f32> = peaks.iter().take(n_formants).map(|&(f, _)| f).collect();

    // Sort ascending by frequency (F1 < F2 < F3 < F4).
    formants.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    formants
}

// ---------------------------------------------------------------------------
// Internal: Peaking EQ biquad
// ---------------------------------------------------------------------------

/// Apply a second-order peaking EQ filter to an audio buffer in-place.
///
/// Based on Robert Bristow-Johnson's Audio EQ Cookbook.
/// H(z) has a peak (or notch) at `center_hz` with width `bandwidth_hz`
/// and gain `gain_db`.
fn apply_peaking_eq(
    audio: &mut [f32],
    center_hz: f32,
    bandwidth_hz: f32,
    gain_db: f32,
    sample_rate: f32,
) {
    if audio.is_empty() || gain_db.abs() < 1e-4 {
        return;
    }

    let sr = f64::from(sample_rate);
    let fc = f64::from(center_hz).clamp(20.0, sr / 2.0 - 1.0);
    let bw = f64::from(bandwidth_hz).max(10.0);
    let a_lin = 10.0f64.powf(f64::from(gain_db) / 40.0); // sqrt(linear gain)

    let w0 = 2.0 * std::f64::consts::PI * fc / sr;
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();

    // Q from bandwidth: Q = fc / bw.
    let q = fc / bw;
    let alpha = sin_w0 / (2.0 * q);

    // Peaking EQ coefficients (Bristow-Johnson).
    let b0 = 1.0 + alpha * a_lin;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a_lin;
    let a0 = 1.0 + alpha / a_lin;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha / a_lin;

    // Normalize.
    let inv_a0 = 1.0 / a0;
    let b0 = (b0 * inv_a0) as f32;
    let b1 = (b1 * inv_a0) as f32;
    let b2 = (b2 * inv_a0) as f32;
    let a1 = (a1 * inv_a0) as f32;
    let a2 = (a2 * inv_a0) as f32;

    // Direct Form II Transposed.
    let mut z1: f32 = 0.0;
    let mut z2: f32 = 0.0;

    for sample in audio.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            z1 = 0.0;
            z2 = 0.0;
            continue;
        }
        let x = *sample;
        let y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;

        *sample = if y.is_finite() { y } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, n_samples: usize) -> Vec<f32> {
        let sr = KOKORO_SAMPLE_RATE as f32;
        (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len().max(1) as f32).sqrt()
    }

    // -- Config validation --

    #[test]
    fn test_config_default_valid() {
        FormantTuneConfig::default()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_invalid_n_formants() {
        assert!(FormantTuneConfig::new()
            .with_n_formants(0)
            .validate()
            .is_err());
        assert!(FormantTuneConfig::new()
            .with_n_formants(5)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_blend() {
        assert!(FormantTuneConfig::new()
            .with_blend_amount(-0.1)
            .validate()
            .is_err());
        assert!(FormantTuneConfig::new()
            .with_blend_amount(1.1)
            .validate()
            .is_err());
        assert!(FormantTuneConfig::new()
            .with_blend_amount(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_lpc_order() {
        assert!(FormantTuneConfig::new()
            .with_lpc_order(5)
            .validate()
            .is_err());
        assert!(FormantTuneConfig::new()
            .with_lpc_order(21)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_frame_size() {
        assert!(FormantTuneConfig::new()
            .with_frame_size(32)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_hop_size() {
        assert!(FormantTuneConfig::new()
            .with_hop_size(0)
            .validate()
            .is_err());
        assert!(FormantTuneConfig::new()
            .with_hop_size(2048)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        assert!(FormantTuneConfig::new()
            .with_sample_rate(0.0)
            .validate()
            .is_err());
        assert!(FormantTuneConfig::new()
            .with_sample_rate(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_formant_band_shift() {
        let bands = vec![FormantBand::new().with_shift_hz(600.0)];
        let cfg = FormantTuneConfig::new().with_voice_formants(vec![bands]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_formant_band_bw() {
        let bands = vec![FormantBand::new().with_bandwidth_scale(0.1)];
        let cfg = FormantTuneConfig::new().with_voice_formants(vec![bands]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_formant_band_gain() {
        let bands = vec![FormantBand::new().with_gain_db(15.0)];
        let cfg = FormantTuneConfig::new().with_voice_formants(vec![bands]);
        assert!(cfg.validate().is_err());
    }

    // -- Presets --

    #[test]
    fn test_presets_validate() {
        FormantTuneConfig::natural().validate().expect("natural");
        FormantTuneConfig::bright().validate().expect("bright");
        FormantTuneConfig::dark().validate().expect("dark");
        FormantTuneConfig::nasal().validate().expect("nasal");
        FormantTuneConfig::chest().validate().expect("chest");
    }

    // -- LPC / formant detection --

    #[test]
    fn test_levinson_durbin_silent_signal() {
        let signal = vec![0.0f32; 256];
        let coeffs = levinson_durbin(&signal, 10);
        assert_eq!(coeffs[0], 1.0);
    }

    #[test]
    fn test_levinson_durbin_returns_correct_length() {
        let signal = sine_wave(440.0, 1024);
        let coeffs = levinson_durbin(&signal, 12);
        assert_eq!(coeffs.len(), 13);
        assert_eq!(coeffs[0], 1.0);
    }

    #[test]
    fn test_detect_formants_returns_sorted_frequencies() {
        // Create a signal with energy at multiple frequencies to give LPC
        // something to work with (simulating vowel-like formant structure).
        let sr = KOKORO_SAMPLE_RATE as f32;
        let n = 2048;
        let audio: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                0.5 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1500.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 2500.0 * t).sin()
            })
            .collect();

        let tuner = FormantTuner::with_defaults().unwrap();
        let formants = tuner.detect_formants(&audio);
        assert!(!formants.is_empty(), "should detect at least one formant");
        // Should be sorted ascending.
        for i in 1..formants.len() {
            assert!(
                formants[i] >= formants[i - 1],
                "formants not sorted: {formants:?}"
            );
        }
        // All in valid range.
        for &f in &formants {
            assert!(
                (MIN_FORMANT_HZ..=MAX_FORMANT_HZ).contains(&f),
                "formant {f} out of range"
            );
        }
    }

    #[test]
    fn test_detect_formants_short_audio_returns_empty() {
        let audio = vec![0.5; 32]; // shorter than frame_size
        let tuner = FormantTuner::with_defaults().unwrap();
        let formants = tuner.detect_formants(&audio);
        assert!(formants.is_empty());
    }

    // -- Processing --

    #[test]
    fn test_process_voice_blend_zero_is_identity() {
        let mut audio = sine_wave(440.0, 4096);
        let original = audio.clone();
        let bands = vec![FormantBand::new().with_gain_db(6.0)];
        let cfg = FormantTuneConfig::new()
            .with_voice_formants(vec![bands])
            .with_blend_amount(0.0);
        let mut tuner = FormantTuner::new(cfg).unwrap();
        tuner.process_voice(&mut audio, 0);
        assert_eq!(audio, original, "blend=0 should be identity");
    }

    #[test]
    fn test_process_voice_modifies_signal() {
        let sr = KOKORO_SAMPLE_RATE as f32;
        let n = 4096;
        let mut audio: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                0.5 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1500.0 * t).sin()
            })
            .collect();
        let original = audio.clone();
        let bands = vec![
            FormantBand::new().with_gain_db(6.0).with_shift_hz(100.0),
            FormantBand::new().with_gain_db(-3.0),
        ];
        let cfg = FormantTuneConfig::new()
            .with_voice_formants(vec![bands])
            .with_blend_amount(1.0);
        let mut tuner = FormantTuner::new(cfg).unwrap();
        tuner.process_voice(&mut audio, 0);

        // Should differ from original.
        let diff: f32 = audio
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n as f32;
        assert!(
            diff > 1e-4,
            "processed signal should differ, mean_diff={diff}"
        );
    }

    #[test]
    fn test_process_voice_all_outputs_finite() {
        let sr = KOKORO_SAMPLE_RATE as f32;
        let n = 2048;
        let mut audio: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * std::f32::consts::PI * 300.0 * t).sin()
            })
            .collect();
        // Inject NaN.
        audio[100] = f32::NAN;
        audio[101] = f32::INFINITY;

        let bands = vec![FormantBand::new().with_gain_db(10.0)];
        let cfg = FormantTuneConfig::new()
            .with_voice_formants(vec![bands])
            .with_blend_amount(1.0);
        let mut tuner = FormantTuner::new(cfg).unwrap();
        tuner.process_voice(&mut audio, 0);

        for (i, &s) in audio.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} is non-finite: {s}");
        }
    }

    #[test]
    fn test_process_voice_no_bands_is_identity() {
        let mut audio = sine_wave(440.0, 2048);
        let original = audio.clone();
        // No voice_formants configured.
        let cfg = FormantTuneConfig::new().with_blend_amount(1.0);
        let mut tuner = FormantTuner::new(cfg).unwrap();
        tuner.process_voice(&mut audio, 0);
        assert_eq!(audio, original, "no bands should be identity");
    }

    #[test]
    fn test_peaking_eq_zero_gain_is_identity() {
        let mut audio = sine_wave(440.0, 1024);
        let original = audio.clone();
        apply_peaking_eq(&mut audio, 1000.0, 200.0, 0.0, KOKORO_SAMPLE_RATE as f32);
        assert_eq!(audio, original);
    }

    #[test]
    fn test_peaking_eq_positive_gain_boosts() {
        let mut audio = sine_wave(1000.0, 4096);
        let dry_rms = rms(&audio);
        apply_peaking_eq(&mut audio, 1000.0, 200.0, 6.0, KOKORO_SAMPLE_RATE as f32);
        let wet_rms = rms(&audio);
        assert!(
            wet_rms > dry_rms,
            "positive gain should boost: dry={dry_rms}, wet={wet_rms}"
        );
    }

    #[test]
    fn test_formant_tuner_reset() {
        let mut tuner = FormantTuner::with_defaults().unwrap();
        tuner.reset(); // Should not panic.
    }
}
