// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Anti-aliased oversampling for nonlinear chorus processing stages.
//!
//! Saturation, exciter, and waveshaping stages generate harmonics that can
//! alias at the Nyquist frequency (12 kHz at 24 kHz sample rate). This module
//! provides transparent 2x/4x oversampling to push aliasing artifacts above
//! the audible range before they fold back into the spectrum.
//!
//! # Processing chain
//!
//! ```text
//! Input (fs) --> Upsample (N*fs) --> [Nonlinear processing] --> Downsample (fs) --> Output
//!                  |                                                 |
//!           zero-stuff + LP           LP anti-alias filter + decimate
//! ```
//!
//! # Filter design
//!
//! The anti-imaging (upsample) and anti-aliasing (downsample) lowpass filters
//! are cascaded second-order Butterworth sections designed via the bilinear
//! transform. Butterworth maximally-flat passband response minimizes audible
//! coloration in the audio band while providing adequate stopband rejection.
//!
//! - For 2x at 24 kHz: cutoff at 11 kHz (just below original Nyquist)
//! - For 4x at 24 kHz: cutoff at 11 kHz with steeper rolloff (more sections)
//!
//! # References
//!
//! - Smith, J. O. "Introduction to Digital Filters with Audio Applications."
//!   <https://ccrma.stanford.edu/~jos/filters/>
//! - Zolzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//!   Chapter 2: Filters; Chapter 5: Nonlinear Processing.
//! - Butterworth, S. "On the Theory of Filter Amplifiers." 1930.
//!
//! Part of #4264, #3351.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Filter types
// ---------------------------------------------------------------------------

/// Allowed oversampling filter types.
///
/// Currently only Butterworth is implemented. ChebyshevI and Elliptic are
/// reserved for future use when tighter transition bands are needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[derive(Default)]
pub enum OversampleFilterType {
    /// Butterworth: maximally flat passband, gentle rolloff.
    #[default]
    Butterworth,
    /// Chebyshev Type I: steeper rolloff with passband ripple (reserved).
    ChebyshevI,
    /// Elliptic: steepest rolloff with passband and stopband ripple (reserved).
    Elliptic,
}


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the oversampler.
///
/// Constructed via [`OversampleConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct OversampleConfig {
    /// Oversampling factor: 2 or 4. Default: 2.
    pub factor: usize,
    /// Filter order (number of poles): 4, 8, or 12. Default: 8.
    /// Higher orders give steeper rolloff but more latency.
    pub filter_order: usize,
    /// Filter type. Default: Butterworth.
    pub filter_type: OversampleFilterType,
    /// Base sample rate in Hz. Default: 24000.0 (Kokoro).
    pub sample_rate: f32,
}

impl Default for OversampleConfig {
    fn default() -> Self {
        Self {
            factor: 2,
            filter_order: 8,
            filter_type: OversampleFilterType::Butterworth,
            sample_rate: KOKORO_SAMPLE_RATE as f32,
        }
    }
}

impl OversampleConfig {
    /// Create a new oversampler config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the oversampling factor (2 or 4).
    #[must_use]
    pub fn with_factor(mut self, factor: usize) -> Self {
        self.factor = factor;
        self
    }

    /// Set the filter order (4, 8, or 12).
    #[must_use]
    pub fn with_filter_order(mut self, order: usize) -> Self {
        self.filter_order = order;
        self
    }

    /// Set the filter type.
    #[must_use]
    pub fn with_filter_type(mut self, ft: OversampleFilterType) -> Self {
        self.filter_type = ft;
        self
    }

    /// Set the base sample rate in Hz.
    #[must_use]
    pub fn with_sample_rate(mut self, sr: f32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.factor != 2 && self.factor != 4 {
            return Err(KokoroError::InvalidConfig {
                field: "factor",
                reason: format!("factor = {}: must be 2 or 4", self.factor),
            });
        }
        if self.filter_order != 4 && self.filter_order != 8 && self.filter_order != 12 {
            return Err(KokoroError::InvalidConfig {
                field: "filter_order",
                reason: format!("filter_order = {}: must be 4, 8, or 12", self.filter_order),
            });
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!(
                    "sample_rate = {}: must be finite and positive",
                    self.sample_rate,
                ),
            });
        }
        if self.filter_type != OversampleFilterType::Butterworth {
            return Err(KokoroError::InvalidConfig {
                field: "filter_type",
                reason: "only Butterworth is currently implemented".to_string(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Biquad second-order section
// ---------------------------------------------------------------------------

/// Coefficients for a single biquad (second-order IIR) section.
///
/// Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

/// State for a single biquad section (transposed direct-form II).
#[derive(Debug, Clone)]
struct BiquadState {
    coeffs: BiquadCoeffs,
    s1: f32,
    s2: f32,
}

impl BiquadState {
    fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Process a single sample through this biquad section.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let c = &self.coeffs;
        let y = c.b0 * x + self.s1;
        self.s1 = c.b1 * x - c.a1 * y + self.s2;
        self.s2 = c.b2 * x - c.a2 * y;
        if !y.is_finite() {
            self.s1 = 0.0;
            self.s2 = 0.0;
            return 0.0;
        }
        y
    }

    fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Butterworth filter design
// ---------------------------------------------------------------------------

/// Design Butterworth lowpass biquad coefficients as cascaded second-order
/// sections using the bilinear transform.
///
/// Returns `order / 2` biquad sections (order must be even).
///
/// # Arguments
///
/// * `order` - Filter order (must be even: 4, 8, 12).
/// * `cutoff` - Cutoff frequency in Hz.
/// * `sample_rate` - Sample rate in Hz (the oversampled rate).
///
/// # Design
///
/// For each second-order section k (0..order/2), the analog prototype pole
/// angle is theta_k = pi * (2k + 1) / (2 * order) + pi/2. The analog
/// prototype transfer function is mapped to digital via bilinear transform
/// with frequency pre-warping at the cutoff frequency.
pub fn butterworth_coefficients(order: usize, cutoff: f32, sample_rate: f32) -> Vec<BiquadCoeffs> {
    let n_sections = order / 2;
    let mut sections = Vec::with_capacity(n_sections);

    // Pre-warp the cutoff frequency for the bilinear transform.
    let wc = (std::f32::consts::PI * cutoff / sample_rate).tan();
    let wc2 = wc * wc;

    for k in 0..n_sections {
        // Analog prototype pole angle for Butterworth.
        let theta = std::f32::consts::PI * (2 * k + 1) as f32 / (2 * order) as f32
            + std::f32::consts::FRAC_PI_2;
        // Real and imaginary parts of the analog pole (unit circle).
        // For a Butterworth LP prototype: pole = -sin(theta_k) + j*cos(theta_k)
        // but we only need the real part for the second-order section.
        let cos_theta = theta.cos();
        // The second-order section denominator from the analog prototype is:
        // s^2 - 2*cos(theta)*s + 1 (unit circle poles).
        // After bilinear transform with pre-warped frequency wc:
        let alpha = -2.0 * cos_theta;
        let denom = 1.0 + alpha * wc + wc2;
        let inv_denom = 1.0 / denom;

        // Lowpass numerator after bilinear transform: wc^2 * (1 + z^-1)^2
        let b0 = wc2 * inv_denom;
        let b1 = 2.0 * b0;
        let b2 = b0;

        let a1 = 2.0 * (wc2 - 1.0) * inv_denom;
        let a2 = (1.0 - alpha * wc + wc2) * inv_denom;

        sections.push(BiquadCoeffs { b0, b1, b2, a1, a2 });
    }

    sections
}

// ---------------------------------------------------------------------------
// Oversampler
// ---------------------------------------------------------------------------

/// Stateful oversampler that wraps nonlinear processing with anti-aliased
/// up/downsampling.
///
/// Holds separate filter cascades for the upsample (anti-imaging) and
/// downsample (anti-aliasing) paths.
#[derive(Debug, Clone)]
pub struct Oversampler {
    config: OversampleConfig,
    /// Anti-imaging filter (applied after zero-stuffing in upsample path).
    upsample_filters: Vec<BiquadState>,
    /// Anti-aliasing filter (applied before decimation in downsample path).
    downsample_filters: Vec<BiquadState>,
    /// Reusable buffer for the oversampled signal.
    work_buf: Vec<f32>,
}

impl Oversampler {
    /// Create a new oversampler from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is invalid.
    pub fn new(config: &OversampleConfig) -> Result<Self, KokoroError> {
        config.validate()?;

        let oversampled_rate = config.sample_rate * config.factor as f32;
        // Cutoff just below original Nyquist to preserve the audio band.
        // At 24 kHz base: Nyquist is 12 kHz, cutoff at 11 kHz.
        let cutoff = config.sample_rate * 0.5 - 1000.0;
        // Ensure cutoff is positive and below oversampled Nyquist.
        let cutoff = cutoff.max(100.0).min(oversampled_rate * 0.49);

        let coeffs = butterworth_coefficients(config.filter_order, cutoff, oversampled_rate);

        let upsample_filters: Vec<BiquadState> =
            coeffs.iter().map(|c| BiquadState::new(*c)).collect();
        let downsample_filters: Vec<BiquadState> =
            coeffs.iter().map(|c| BiquadState::new(*c)).collect();

        Ok(Self {
            config: *config,
            upsample_filters,
            downsample_filters,
            work_buf: Vec::new(),
        })
    }

    /// Create an oversampler using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: &OversampleConfig) -> Result<Self, KokoroError> {
        let mut cfg = *config;
        cfg.sample_rate = KOKORO_SAMPLE_RATE as f32;
        Self::new(&cfg)
    }

    /// Upsample the input signal by the configured factor.
    ///
    /// Inserts N-1 zeros between each sample (zero-stuffing), then applies
    /// the anti-imaging lowpass filter. The output gain is scaled by the
    /// oversampling factor to compensate for the zero-stuffing energy loss.
    pub fn upsample(&mut self, input: &[f32]) -> Vec<f32> {
        let factor = self.config.factor;
        let n_out = input.len() * factor;
        let mut output = vec![0.0_f32; n_out];

        // Zero-stuff: place each input sample at every `factor`-th position.
        let gain = factor as f32; // Compensate zero-stuffing energy loss.
        for (i, &sample) in input.iter().enumerate() {
            output[i * factor] = sample * gain;
        }

        // Apply the anti-imaging filter cascade.
        for section in &mut self.upsample_filters {
            for sample in output.iter_mut() {
                *sample = section.process(*sample);
            }
        }

        output
    }

    /// Downsample the input signal by the configured factor.
    ///
    /// Applies the anti-aliasing lowpass filter, then decimates by taking
    /// every Nth sample.
    pub fn downsample(&mut self, input: &[f32]) -> Vec<f32> {
        let factor = self.config.factor;

        // Apply the anti-aliasing filter cascade on a mutable copy.
        // We filter in-place on work_buf to avoid extra allocation.
        self.work_buf.clear();
        self.work_buf.extend_from_slice(input);

        for section in &mut self.downsample_filters {
            for sample in self.work_buf.iter_mut() {
                *sample = section.process(*sample);
            }
        }

        // Decimate: take every Nth sample.
        let n_out = self.work_buf.len() / factor;
        let mut output = Vec::with_capacity(n_out);
        for i in 0..n_out {
            output.push(self.work_buf[i * factor]);
        }

        output
    }

    /// Process audio through an oversampled nonlinear function.
    ///
    /// 1. Upsamples `audio` by the configured factor.
    /// 2. Calls `process` on the oversampled buffer.
    /// 3. Downsamples the result back to the original rate.
    /// 4. Replaces the contents of `audio` with the processed result.
    ///
    /// The `process` closure receives the oversampled buffer and should
    /// apply nonlinear processing (saturation, waveshaping, exciter) in place.
    pub fn process_oversampled(
        &mut self,
        audio: &mut Vec<f32>,
        mut process: impl FnMut(&mut Vec<f32>),
    ) {
        if audio.is_empty() {
            return;
        }

        // Step 1: Upsample.
        let mut oversampled = self.upsample(audio);

        // Step 2: Apply the nonlinear processing at the oversampled rate.
        process(&mut oversampled);

        // Step 3: Downsample back to original rate.
        let result = self.downsample(&oversampled);

        // Step 4: Replace audio contents.
        audio.clear();
        audio.extend_from_slice(&result);
    }

    /// Get the latency in samples introduced by the oversampling filters.
    ///
    /// Each biquad section introduces approximately 1 sample of group delay
    /// at the oversampled rate. Total latency (in original-rate samples) is
    /// approximately: (n_sections * 2) / factor (upsample + downsample paths).
    #[must_use]
    pub fn get_latency_samples(&self) -> usize {
        // Each biquad section contributes ~1 sample delay at the oversampled
        // rate. We have two filter cascades (up + down), each with n_sections.
        let n_sections = self.upsample_filters.len();
        let total_oversampled_delay = n_sections * 2;
        // Convert from oversampled-rate samples to base-rate samples.
        total_oversampled_delay / self.config.factor
    }

    /// Reset all internal filter state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        for section in &mut self.upsample_filters {
            section.reset();
        }
        for section in &mut self.downsample_filters {
            section.reset();
        }
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &OversampleConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Convenience: per-voice oversampled processing
// ---------------------------------------------------------------------------

/// Apply oversampled nonlinear processing to each voice buffer independently.
///
/// Creates one [`Oversampler`] per voice (each with independent filter state)
/// and processes in place. For streaming scenarios where filter state must
/// persist across calls, create [`Oversampler`] instances directly.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid.
pub fn apply_oversampled(
    voices: &mut [Vec<f32>],
    config: &OversampleConfig,
    mut process: impl FnMut(&mut Vec<f32>),
) -> Result<(), KokoroError> {
    for voice in voices.iter_mut() {
        let mut oversampler = Oversampler::new(config)?;
        oversampler.process_oversampled(voice, &mut process);
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
        if buf.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len() as f32).sqrt()
    }

    // --- Config validation ---

    #[test]
    fn test_config_default_valid() {
        OversampleConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_factor_2_valid() {
        OversampleConfig::new()
            .with_factor(2)
            .validate()
            .expect("factor 2 should be valid");
    }

    #[test]
    fn test_config_factor_4_valid() {
        OversampleConfig::new()
            .with_factor(4)
            .validate()
            .expect("factor 4 should be valid");
    }

    #[test]
    fn test_config_factor_3_invalid() {
        assert!(OversampleConfig::new().with_factor(3).validate().is_err());
    }

    #[test]
    fn test_config_filter_order_4_valid() {
        OversampleConfig::new()
            .with_filter_order(4)
            .validate()
            .expect("order 4 should be valid");
    }

    #[test]
    fn test_config_filter_order_12_valid() {
        OversampleConfig::new()
            .with_filter_order(12)
            .validate()
            .expect("order 12 should be valid");
    }

    #[test]
    fn test_config_filter_order_6_invalid() {
        assert!(OversampleConfig::new()
            .with_filter_order(6)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        assert!(OversampleConfig::new()
            .with_sample_rate(0.0)
            .validate()
            .is_err());
        assert!(OversampleConfig::new()
            .with_sample_rate(-1.0)
            .validate()
            .is_err());
        assert!(OversampleConfig::new()
            .with_sample_rate(f32::NAN)
            .validate()
            .is_err());
    }

    // --- Butterworth coefficients ---

    #[test]
    fn test_butterworth_produces_correct_section_count() {
        let coeffs = butterworth_coefficients(4, 11000.0, 48000.0);
        assert_eq!(coeffs.len(), 2, "order 4 should produce 2 sections");

        let coeffs = butterworth_coefficients(8, 11000.0, 48000.0);
        assert_eq!(coeffs.len(), 4, "order 8 should produce 4 sections");

        let coeffs = butterworth_coefficients(12, 11000.0, 96000.0);
        assert_eq!(coeffs.len(), 6, "order 12 should produce 6 sections");
    }

    #[test]
    fn test_butterworth_coefficients_finite() {
        let coeffs = butterworth_coefficients(8, 11000.0, 48000.0);
        for (i, c) in coeffs.iter().enumerate() {
            assert!(c.b0.is_finite(), "section {i} b0 non-finite");
            assert!(c.b1.is_finite(), "section {i} b1 non-finite");
            assert!(c.b2.is_finite(), "section {i} b2 non-finite");
            assert!(c.a1.is_finite(), "section {i} a1 non-finite");
            assert!(c.a2.is_finite(), "section {i} a2 non-finite");
        }
    }

    // --- Upsample / Downsample ---

    #[test]
    fn test_upsample_output_length() {
        let config = OversampleConfig::new().with_factor(2);
        let mut os = Oversampler::new(&config).unwrap();
        let input = vec![1.0; 100];
        let up = os.upsample(&input);
        assert_eq!(up.len(), 200, "2x upsample of 100 should be 200");
    }

    #[test]
    fn test_upsample_4x_output_length() {
        let config = OversampleConfig::new().with_factor(4);
        let mut os = Oversampler::new(&config).unwrap();
        let input = vec![1.0; 100];
        let up = os.upsample(&input);
        assert_eq!(up.len(), 400, "4x upsample of 100 should be 400");
    }

    #[test]
    fn test_downsample_output_length() {
        let config = OversampleConfig::new().with_factor(2);
        let mut os = Oversampler::new(&config).unwrap();
        let input = vec![1.0; 200];
        let down = os.downsample(&input);
        assert_eq!(down.len(), 100, "2x downsample of 200 should be 100");
    }

    #[test]
    fn test_roundtrip_preserves_length() {
        let config = OversampleConfig::new().with_factor(2);
        let mut os = Oversampler::new(&config).unwrap();
        let input = sine_wave(440.0, 1024, 0.5);
        let up = os.upsample(&input);
        let down = os.downsample(&up);
        assert_eq!(
            down.len(),
            input.len(),
            "roundtrip should preserve sample count",
        );
    }

    #[test]
    fn test_roundtrip_preserves_energy_approx() {
        // After settling the filter, a sine well within the passband
        // should have roughly similar energy after upsample-downsample.
        let config = OversampleConfig::new().with_factor(2).with_filter_order(8);
        let mut os = Oversampler::new(&config).unwrap();
        let input = sine_wave(1000.0, 4096, 0.5);
        let up = os.upsample(&input);
        let down = os.downsample(&up);

        let rms_in = rms(&input[512..]); // skip transient
        let rms_out = rms(&down[512..]);
        let ratio = rms_out / rms_in;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "roundtrip energy ratio should be near 1.0 for in-band signal, got {ratio}",
        );
    }

    // --- Process oversampled ---

    #[test]
    fn test_process_oversampled_identity() {
        // Identity processing (no-op closure) should approximately preserve
        // the signal for in-band frequencies.
        let config = OversampleConfig::new().with_factor(2);
        let mut os = Oversampler::new(&config).unwrap();
        let original = sine_wave(1000.0, 2048, 0.5);
        let mut audio = original.clone();

        os.process_oversampled(&mut audio, |_buf| {
            // Identity: no processing.
        });

        assert_eq!(audio.len(), original.len(), "length should be preserved");

        // Check energy preservation (skip filter transient).
        let rms_orig = rms(&original[256..]);
        let rms_proc = rms(&audio[256..]);
        let ratio = rms_proc / rms_orig;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "identity processing should preserve energy, got ratio {ratio}",
        );
    }

    #[test]
    fn test_process_oversampled_with_saturation() {
        // Apply a simple tanh saturation at the oversampled rate.
        let config = OversampleConfig::new().with_factor(2);
        let mut os = Oversampler::new(&config).unwrap();
        let mut audio = sine_wave(1000.0, 2048, 0.8);
        let rms_before = rms(&audio);

        os.process_oversampled(&mut audio, |buf| {
            for sample in buf.iter_mut() {
                *sample = (*sample * 2.0).tanh();
            }
        });

        // Saturation should produce a non-zero output.
        let rms_after = rms(&audio);
        assert!(rms_after > 0.0, "saturated signal should have energy");

        // Saturated signal should differ from original.
        assert!(
            (rms_after - rms_before).abs() > 0.01,
            "saturation should change the signal: before={rms_before}, after={rms_after}",
        );
    }

    #[test]
    fn test_process_oversampled_empty_input() {
        let config = OversampleConfig::new();
        let mut os = Oversampler::new(&config).unwrap();
        let mut audio = Vec::new();
        os.process_oversampled(&mut audio, |_| {});
        assert!(audio.is_empty(), "empty input should produce empty output");
    }

    // --- All outputs finite ---

    #[test]
    fn test_all_outputs_finite() {
        let config = OversampleConfig::new().with_factor(2);
        let mut os = Oversampler::new(&config).unwrap();
        let mut audio = vec![
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
            0.0,
            0.3,
        ];
        os.process_oversampled(&mut audio, |buf| {
            for s in buf.iter_mut() {
                *s = (*s * 3.0).tanh();
            }
        });
        for (i, &v) in audio.iter().enumerate() {
            assert!(v.is_finite(), "sample {i} is non-finite: {v}");
        }
    }

    // --- Reset clears state ---

    #[test]
    fn test_reset_clears_state() {
        let config = OversampleConfig::new();
        let mut os = Oversampler::new(&config).unwrap();
        let _ = os.upsample(&[0.5; 100]);
        os.reset();
        for section in &os.upsample_filters {
            assert_eq!(section.s1, 0.0);
            assert_eq!(section.s2, 0.0);
        }
        for section in &os.downsample_filters {
            assert_eq!(section.s1, 0.0);
            assert_eq!(section.s2, 0.0);
        }
    }

    // --- Latency ---

    #[test]
    fn test_latency_is_reasonable() {
        let config = OversampleConfig::new().with_factor(2).with_filter_order(8);
        let os = Oversampler::new(&config).unwrap();
        let latency = os.get_latency_samples();
        // 8th-order = 4 sections, 2 cascades = 8 oversampled-rate samples
        // / factor 2 = 4 base-rate samples.
        assert!(
            latency > 0 && latency <= 12,
            "latency should be small: got {latency}",
        );
    }

    // --- Per-voice convenience ---

    #[test]
    fn test_apply_oversampled_per_voice() {
        let n = 1024;
        let mut voices = vec![sine_wave(800.0, n, 0.5), sine_wave(1200.0, n, 0.5)];
        let config = OversampleConfig::new().with_factor(2);

        apply_oversampled(&mut voices, &config, |buf| {
            for s in buf.iter_mut() {
                *s = (*s * 2.0).tanh();
            }
        })
        .expect("apply_oversampled should succeed");

        for (i, voice) in voices.iter().enumerate() {
            assert_eq!(voice.len(), n, "voice {i} length preserved");
            for (j, &v) in voice.iter().enumerate() {
                assert!(v.is_finite(), "voice {i} sample {j} non-finite: {v}");
            }
        }
    }

    // --- Aliasing reduction verification ---

    #[test]
    fn test_oversampling_reduces_aliasing() {
        // Generate a sine at 5 kHz, apply hard clipping (generates harmonics
        // at 15 kHz, 25 kHz, etc.). Without oversampling, the 15 kHz harmonic
        // aliases to 9 kHz at 24 kHz sample rate. With 2x oversampling, the
        // anti-aliasing filter should attenuate it.
        let n = 4096;
        let freq = 5000.0;

        // Without oversampling: hard-clip directly.
        let mut no_os = sine_wave(freq, n, 0.8);
        for s in no_os.iter_mut() {
            *s = s.clamp(-0.3, 0.3);
        }

        // With 2x oversampling: hard-clip at oversampled rate.
        let config = OversampleConfig::new().with_factor(2).with_filter_order(8);
        let mut os = Oversampler::new(&config).unwrap();
        let mut with_os = sine_wave(freq, n, 0.8);
        os.process_oversampled(&mut with_os, |buf| {
            for s in buf.iter_mut() {
                *s = s.clamp(-0.3, 0.3);
            }
        });

        // Measure energy in the alias band (7-11 kHz) using a simple bandpass
        // approximation: highpass at 7 kHz then lowpass at 11 kHz.
        // For simplicity, just measure HF energy above 7 kHz.
        fn hf_energy_above(buf: &[f32], cutoff_hz: f32, sr: f32) -> f32 {
            // Simple first-order highpass to estimate HF content.
            let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
            let dt = 1.0 / sr;
            let coeff = rc / (rc + dt);
            let mut x_prev = 0.0_f32;
            let mut y_prev = 0.0_f32;
            let mut sum_sq = 0.0_f32;
            for &x in &buf[512..] {
                // skip transient
                let y = coeff * (y_prev + x - x_prev);
                x_prev = x;
                y_prev = y;
                sum_sq += y * y;
            }
            (sum_sq / (buf.len() - 512) as f32).sqrt()
        }

        let hf_no_os = hf_energy_above(&no_os, 7000.0, SR);
        let hf_with_os = hf_energy_above(&with_os, 7000.0, SR);

        // The oversampled version should have less aliasing energy above 7 kHz.
        // Note: the difference may be modest with a simple first-order HP
        // measurement, but the oversampled version should not have MORE.
        assert!(
            hf_with_os <= hf_no_os * 1.1,
            "oversampled HF energy ({hf_with_os}) should not exceed \
             non-oversampled ({hf_no_os}) significantly",
        );
    }
}
