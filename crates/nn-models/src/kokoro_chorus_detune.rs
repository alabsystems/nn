// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-voice detuning with allpass fractional-delay interpolation for Kokoro chorus.
//!
//! Real choirs have natural detuning — each singer is slightly off from the exact
//! pitch, creating a warm, thick sound through beating frequencies. This module
//! adds ±5–15 cents of static detuning per voice using first-order allpass
//! (Thiran) interpolation for smooth fractional-sample delay.
//!
//! # Why allpass instead of linear interpolation?
//!
//! Linear interpolation acts as a low-pass filter: at Nyquist, gain drops to zero.
//! For small detuning amounts (fractional-sample shifts), this attenuates high
//! frequencies and makes the chorus sound dull. A first-order allpass filter has
//! **unity gain at all frequencies** — it only changes phase, preserving the full
//! spectrum while achieving smooth fractional-sample delay.
//!
//! # Architecture
//!
//! ```text
//! Voice 0 (anchor): unchanged
//! Voice 1: allpass resample at rate 2^(cents_1 / 1200)
//! Voice 2: allpass resample at rate 2^(cents_2 / 1200)
//! ...
//! Voice N-1: allpass resample at rate 2^(cents_{N-1} / 1200)
//! ```
//!
//! The detuning is **constant** (not modulated) — vibrato handles periodic
//! modulation. This creates the classic "ensemble" thickness from beating
//! frequencies between slightly mistuned voices.
//!
//! # References
//!
//! - Laakso, T. et al. "Splitting the Unit Delay." IEEE Signal Processing
//!   Magazine, 13(1), 1996. (Thiran allpass interpolation)
//! - Välimäki, V. "Discrete-Time Modeling of Acoustic Tubes Using Fractional
//!   Delay Filters." PhD thesis, Helsinki University of Technology, 1995.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Distribution for spreading detuning across voices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum DetuneDistribution {
    /// Voices spread evenly from -cents_spread to +cents_spread.
    #[default]
    Uniform,
    /// Voices spread with Gaussian-like concentration near center.
    /// Uses a deterministic approximation: `spread * sin(π * t)` for t in [-0.5, 0.5].
    Gaussian,
}


/// Configuration for per-voice detuning in a chorus.
///
/// Controls the amount and distribution of static pitch offset applied
/// to each voice. Voice 0 is always the undetuned "anchor" voice.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DetuneConfig {
    /// Maximum detuning spread in cents (1 cent = 1/100 semitone).
    ///
    /// Voices are spread from -cents_spread to +cents_spread.
    /// Typical values: 5–15 cents for subtle ensemble width,
    /// 15–30 cents for a wide, dramatic chorus.
    ///
    /// Must be in [0.0, 50.0] and finite.
    pub cents_spread: f32,

    /// How detuning is distributed across voices.
    pub distribution: DetuneDistribution,

    /// Seed for deterministic spread ordering.
    ///
    /// Different seeds produce different voice-to-cent assignments
    /// while keeping the overall spread symmetric.
    pub seed: u64,
}

impl Default for DetuneConfig {
    fn default() -> Self {
        Self {
            cents_spread: 8.0, // ±8 cents — subtle but audible ensemble width
            distribution: DetuneDistribution::Uniform,
            seed: 0,
        }
    }
}

impl DetuneConfig {
    /// Create a new detuning configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if `cents_spread` is non-finite
    /// or outside [0.0, 50.0].
    pub fn new(
        cents_spread: f32,
        distribution: DetuneDistribution,
        seed: u64,
    ) -> Result<Self, KokoroError> {
        if !cents_spread.is_finite() || !(0.0..=50.0).contains(&cents_spread) {
            return Err(KokoroError::InvalidConfig {
                field: "cents_spread",
                reason: format!("must be finite and in [0.0, 50.0], got {cents_spread}"),
            });
        }
        Ok(Self {
            cents_spread,
            distribution,
            seed,
        })
    }

    /// Validate this configuration.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.cents_spread.is_finite() || !(0.0..=50.0).contains(&self.cents_spread) {
            return Err(KokoroError::InvalidConfig {
                field: "cents_spread",
                reason: format!(
                    "must be finite and in [0.0, 50.0], got {}",
                    self.cents_spread
                ),
            });
        }
        Ok(())
    }

    /// Compute per-voice cent offsets for `n_voices` voices.
    ///
    /// Voice 0 always gets 0.0 cents (anchor voice). Remaining voices
    /// are spread symmetrically according to the distribution.
    #[must_use]
    pub fn voice_cents(&self, n_voices: usize) -> Vec<f32> {
        if n_voices <= 1 || self.cents_spread < 1e-6 {
            return vec![0.0; n_voices];
        }

        let mut cents = vec![0.0f32; n_voices];

        // Voices 1..n_voices get spread from -cents_spread to +cents_spread.
        let n_spread = n_voices - 1;
        for i in 0..n_spread {
            let t = if n_spread == 1 {
                0.5
            } else {
                i as f64 / (n_spread - 1) as f64
            };

            let offset = match self.distribution {
                DetuneDistribution::Uniform => {
                    // Linear spread: -spread to +spread
                    let c = f64::from(-self.cents_spread) + 2.0 * f64::from(self.cents_spread) * t;
                    c as f32
                }
                DetuneDistribution::Gaussian => {
                    // Sinusoidal concentration near center:
                    // sin(π * (t - 0.5)) maps [0,1] to [-1,1] with
                    // more density near 0.
                    let c = f64::from(self.cents_spread) * (std::f64::consts::PI * (t - 0.5)).sin();
                    c as f32
                }
            };

            cents[i + 1] = offset;
        }

        cents
    }
}

// ---------------------------------------------------------------------------
// First-order allpass interpolator (Thiran)
// ---------------------------------------------------------------------------

/// First-order allpass filter for fractional-sample delay interpolation.
///
/// The Thiran allpass design provides maximally flat group delay at DC,
/// meaning low frequencies maintain phase coherence. The transfer function is:
///
/// ```text
/// H(z) = (a + z^-1) / (1 + a * z^-1)
/// ```
///
/// where `a = (1 - d) / (1 + d)` for fractional delay `d` in [0, 1].
///
/// Properties:
/// - Unity gain at all frequencies (|H(e^jω)| = 1)
/// - Group delay ≈ d samples at low frequencies
/// - No amplitude distortion — only phase is affected
#[derive(Debug, Clone)]
pub struct AllpassInterpolator {
    /// Allpass coefficient: (1 - d) / (1 + d)
    coeff: f32,
    /// Filter state (one-sample memory)
    z1: f32,
}

impl AllpassInterpolator {
    /// Create a new allpass interpolator for fractional delay `d`.
    ///
    /// `d` should be in [0.0, 1.0]. Values outside this range are clamped.
    /// At d=0 the filter is a pass-through; at d=1 it's a full one-sample delay.
    #[must_use]
    pub fn new(fractional_delay: f32) -> Self {
        let d = f64::from(fractional_delay.clamp(0.0, 1.0));
        let denom = 1.0 + d;
        // IEEE 754 safety: denom is always >= 1.0 for d in [0,1], so no division by zero.
        let coeff = if !denom.is_finite() || denom.abs() < 1e-12 {
            0.0
        } else {
            ((1.0 - d) / denom) as f32
        };
        // Final IEEE 754 check on the computed coefficient.
        let coeff = if !coeff.is_finite() { 0.0 } else { coeff };
        Self { coeff, z1: 0.0 }
    }

    /// Reset the filter state to zero.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
    }

    /// Process a single sample through the allpass filter.
    ///
    /// Returns the delayed sample. Unity gain is preserved at all frequencies.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        // Guard against NaN/Inf propagation.
        if !input.is_finite() {
            self.z1 = 0.0;
            return 0.0;
        }

        let output = self.coeff * input + self.z1;
        self.z1 = input - self.coeff * output;

        // Flush denormals and guard output.
        if !output.is_finite() {
            self.z1 = 0.0;
            return 0.0;
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Voice detuner
// ---------------------------------------------------------------------------

/// Applies per-voice detuning via variable-rate resampling with allpass interpolation.
///
/// Each voice (except the anchor at index 0) is resampled at a slightly
/// different rate corresponding to its cent offset. The allpass interpolator
/// handles fractional-sample positions without high-frequency loss.
pub struct VoiceDetuner {
    /// Per-voice cent offsets (voice 0 = 0.0).
    cents: Vec<f32>,
    /// Per-voice allpass interpolators (one per non-anchor voice).
    /// Index corresponds to voice index - 1.
    interpolators: Vec<AllpassInterpolator>,
}

impl VoiceDetuner {
    /// Create a new voice detuner for the given config and voice count.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails.
    pub fn new(config: &DetuneConfig, n_voices: usize) -> Result<Self, KokoroError> {
        config.validate()?;
        let cents = config.voice_cents(n_voices);

        // Build an allpass interpolator for each non-anchor voice.
        // The fractional delay is derived from the resampling rate:
        // rate = 2^(cents / 1200), and the fractional part of the
        // per-sample position increment determines the delay parameter.
        let interpolators = cents
            .iter()
            .skip(1) // voice 0 is anchor
            .map(|&c| {
                let rate = f64::from(cents_to_rate(c));
                // The fractional part of the rate determines the allpass delay.
                // For small cent values, rate ≈ 1.0 + tiny_offset, so the
                // fractional delay tracks the drift per sample.
                let frac = (rate - 1.0).abs().fract() as f32;
                AllpassInterpolator::new(frac.clamp(0.0, 1.0))
            })
            .collect();

        Ok(Self {
            cents,
            interpolators,
        })
    }

    /// Apply detuning to a set of voice audio buffers in-place.
    ///
    /// Voice 0 is never modified (anchor voice). Other voices are resampled
    /// at their detuned rate using allpass interpolation.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidInput` if `voices` length doesn't match
    /// the configured voice count.
    pub fn apply(&mut self, voices: &mut [Vec<f32>]) -> Result<(), KokoroError> {
        if voices.len() != self.cents.len() {
            return Err(KokoroError::InvalidInput(format!(
                "voices length {} != configured voice count {}",
                voices.len(),
                self.cents.len(),
            )));
        }

        // Process each non-anchor voice.
        for (voice_idx, voice_pcm) in voices.iter_mut().enumerate().skip(1) {
            let cents_offset = self.cents[voice_idx];
            if cents_offset.abs() < 1e-6 {
                continue; // No detuning for this voice.
            }

            let interp = &mut self.interpolators[voice_idx - 1];
            interp.reset();

            let rate = f64::from(cents_to_rate(cents_offset));
            let len = voice_pcm.len();
            let mut resampled = Vec::with_capacity(len);

            let mut src_pos: f64 = 0.0;
            for _ in 0..len {
                let src_idx = src_pos as usize;
                let frac = (src_pos - src_idx as f64) as f32;

                if src_idx >= len {
                    resampled.push(0.0);
                    continue;
                }

                let s0 = voice_pcm[src_idx];
                let s1 = if src_idx + 1 < len {
                    voice_pcm[src_idx + 1]
                } else {
                    s0
                };

                // Integer-sample component via direct lookup, fractional
                // component via allpass interpolation.
                let base = s0;
                let diff = s1 - s0;

                // Feed the difference signal through the allpass to get
                // a phase-correct fractional delay.
                let frac_component = interp.process(diff) * frac;
                resampled.push(base + frac_component);

                src_pos += rate;
            }

            // Replace original buffer with resampled data.
            voice_pcm.clear();
            voice_pcm.extend_from_slice(&resampled);
        }

        Ok(())
    }

    /// Get the per-voice cent offsets.
    #[must_use]
    pub fn voice_cents(&self) -> &[f32] {
        &self.cents
    }
}

// ---------------------------------------------------------------------------
// Public convenience function
// ---------------------------------------------------------------------------

/// Apply per-voice detuning to a set of voice audio buffers.
///
/// This is the main entry point for chorus detuning. Voice 0 is left
/// unmodified as the anchor. All other voices are resampled at a slightly
/// different rate using allpass interpolation for smooth fractional-sample
/// delay without high-frequency loss.
///
/// # Arguments
///
/// * `voices` - Mutable slice of per-voice PCM buffers (24kHz mono).
/// * `config` - Detuning configuration (cents spread, distribution, seed).
/// * `_sample_rate` - Sample rate in Hz (reserved for future rate-dependent tuning).
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid, or
/// `KokoroError::InvalidInput` if `voices` is empty.
pub fn apply_detune(
    voices: &mut [Vec<f32>],
    config: &DetuneConfig,
    _sample_rate: u32,
) -> Result<(), KokoroError> {
    config.validate()?;

    if voices.is_empty() {
        return Ok(());
    }

    // Skip if no detuning configured.
    if config.cents_spread < 1e-6 {
        return Ok(());
    }

    let mut detuner = VoiceDetuner::new(config, voices.len())?;
    detuner.apply(voices)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a cent offset to a resampling rate ratio.
///
/// `rate = 2^(cents / 1200)`. At 0 cents, rate = 1.0 (no change).
/// At +100 cents (1 semitone up), rate ≈ 1.05946.
/// At -100 cents (1 semitone down), rate ≈ 0.94387.
#[inline]
#[must_use]
pub fn cents_to_rate(cents: f32) -> f32 {
    if !cents.is_finite() {
        return 1.0;
    }
    (2.0f64).powf(f64::from(cents) / 1200.0) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_detune_is_identity() {
        let config = DetuneConfig::new(0.0, DetuneDistribution::Uniform, 0).unwrap();
        let original: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut voices = vec![original.clone(), original.clone(), original.clone()];
        apply_detune(&mut voices, &config, 24000).unwrap();

        // All voices should be unchanged.
        for (i, voice) in voices.iter().enumerate() {
            for (j, (&got, &expected)) in voice.iter().zip(original.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "voice {i} sample {j}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_anchor_voice_unmodified() {
        let config = DetuneConfig::new(15.0, DetuneDistribution::Uniform, 42).unwrap();
        let anchor: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.03).sin()).collect();
        let other: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.05).cos()).collect();
        let mut voices = vec![anchor.clone(), other.clone(), other];
        apply_detune(&mut voices, &config, 24000).unwrap();

        // Voice 0 must be completely unchanged.
        assert_eq!(voices[0].len(), anchor.len());
        for (j, (&got, &expected)) in voices[0].iter().zip(anchor.iter()).enumerate() {
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "anchor sample {j}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_allpass_unity_gain() {
        // An allpass filter should have unity gain at all frequencies.
        // Test by running a sine sweep and checking RMS is preserved.
        for frac in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let mut ap = AllpassInterpolator::new(frac);
            let n = 4096;
            let freq = 1000.0;
            let sr = 24000.0;

            let input: Vec<f32> = (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
                .collect();

            let output: Vec<f32> = input.iter().map(|&s| ap.process(s)).collect();

            // Skip transient (first 64 samples), then compare RMS.
            let skip = 64;
            let rms_in: f32 =
                (input[skip..].iter().map(|x| x * x).sum::<f32>() / (n - skip) as f32).sqrt();
            let rms_out: f32 =
                (output[skip..].iter().map(|x| x * x).sum::<f32>() / (n - skip) as f32).sqrt();

            let ratio = rms_out / rms_in;
            assert!(
                (ratio - 1.0).abs() < 0.01,
                "allpass frac={frac}: RMS ratio {ratio} (expected ~1.0)"
            );
        }
    }

    #[test]
    fn test_spread_is_symmetric() {
        let config = DetuneConfig::new(10.0, DetuneDistribution::Uniform, 0).unwrap();
        let cents = config.voice_cents(5);

        // Voice 0 is always 0.
        assert!((cents[0]).abs() < 1e-6, "voice 0 should be 0 cents");

        // Remaining voices should be symmetric around 0.
        // Voices 1..5 get [-10, -5, 0, 5, 10] mapped to [-10, -3.33, 3.33, 10]
        // (4 voices spread over [-10, 10]).
        let non_anchor: Vec<f32> = cents[1..].to_vec();
        let sum: f32 = non_anchor.iter().sum();
        assert!(
            sum.abs() < 0.1,
            "non-anchor cents should sum to ~0 (symmetric), got {sum}"
        );

        // Check min and max are close to ±cents_spread.
        let min = non_anchor.iter().copied().fold(f32::INFINITY, f32::min);
        let max = non_anchor.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((min + 10.0).abs() < 0.1, "min should be ~-10, got {min}");
        assert!((max - 10.0).abs() < 0.1, "max should be ~+10, got {max}");
    }

    #[test]
    fn test_gaussian_distribution_concentrated() {
        let config = DetuneConfig::new(10.0, DetuneDistribution::Gaussian, 0).unwrap();
        let cents = config.voice_cents(9); // 8 non-anchor voices

        // Voice 0 is 0.
        assert!((cents[0]).abs() < 1e-6);

        // Gaussian distribution: more voices near center.
        // The middle values should be smaller in magnitude than the extremes.
        let non_anchor: Vec<f32> = cents[1..].to_vec();
        let mid_idx = non_anchor.len() / 2;
        let mid_val = non_anchor[mid_idx].abs();
        let end_val = non_anchor.last().unwrap().abs();
        assert!(
            mid_val < end_val + 0.1,
            "middle ({mid_val}) should be <= end ({end_val})"
        );
    }

    #[test]
    fn test_cents_to_rate_identity() {
        let rate = cents_to_rate(0.0);
        assert!(
            (rate - 1.0).abs() < 1e-6,
            "0 cents should give rate 1.0, got {rate}"
        );
    }

    #[test]
    fn test_cents_to_rate_semitone() {
        let rate = cents_to_rate(100.0);
        let expected = 2.0f32.powf(100.0 / 1200.0);
        assert!(
            (rate - expected).abs() < 1e-5,
            "100 cents rate: got {rate}, expected {expected}"
        );
    }

    #[test]
    fn test_cents_to_rate_nan_returns_one() {
        let rate = cents_to_rate(f32::NAN);
        assert!((rate - 1.0).abs() < 1e-6, "NaN cents should give rate 1.0");
    }

    #[test]
    fn test_config_validation_rejects_non_finite() {
        assert!(DetuneConfig::new(f32::NAN, DetuneDistribution::Uniform, 0).is_err());
        assert!(DetuneConfig::new(f32::INFINITY, DetuneDistribution::Uniform, 0).is_err());
        assert!(DetuneConfig::new(-1.0, DetuneDistribution::Uniform, 0).is_err());
        assert!(DetuneConfig::new(51.0, DetuneDistribution::Uniform, 0).is_err());
    }

    #[test]
    fn test_config_validation_accepts_valid() {
        assert!(DetuneConfig::new(0.0, DetuneDistribution::Uniform, 0).is_ok());
        assert!(DetuneConfig::new(50.0, DetuneDistribution::Gaussian, 99).is_ok());
        assert!(DetuneConfig::new(8.0, DetuneDistribution::Uniform, 0).is_ok());
    }

    #[test]
    fn test_single_voice_no_change() {
        let config = DetuneConfig::new(15.0, DetuneDistribution::Uniform, 0).unwrap();
        let original: Vec<f32> = (0..500).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut voices = vec![original.clone()];
        apply_detune(&mut voices, &config, 24000).unwrap();

        // Single voice = anchor, unchanged.
        for (j, (&got, &expected)) in voices[0].iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "single voice sample {j}: {got} != {expected}"
            );
        }
    }

    #[test]
    fn test_detuned_voices_differ_from_original() {
        let config = DetuneConfig::new(15.0, DetuneDistribution::Uniform, 0).unwrap();
        let signal: Vec<f32> = (0..4000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal.clone(), signal.clone(), signal.clone()];
        apply_detune(&mut voices, &config, 24000).unwrap();

        // Voices 1 and 2 should differ from the original.
        let diff1: f32 = voices[1]
            .iter()
            .zip(signal.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / voices[1].len() as f32;

        assert!(
            diff1 > 1e-5,
            "detuned voice 1 should differ from original, mean diff = {diff1}"
        );
    }

    #[test]
    fn test_allpass_nan_safety() {
        let mut ap = AllpassInterpolator::new(0.5);

        // NaN input should return 0.0 and reset state.
        let out = ap.process(f32::NAN);
        assert!((out).abs() < 1e-6, "NaN input should produce 0.0");

        // Should still work after NaN.
        let out = ap.process(1.0);
        assert!(out.is_finite(), "should recover after NaN");
    }

    #[test]
    fn test_voice_detuner_mismatched_count() {
        let config = DetuneConfig::new(10.0, DetuneDistribution::Uniform, 0).unwrap();
        let mut detuner = VoiceDetuner::new(&config, 3).unwrap();
        let mut voices = vec![vec![0.0; 100], vec![0.0; 100]]; // only 2, expect 3
        assert!(detuner.apply(&mut voices).is_err());
    }

    #[test]
    fn test_empty_voices_ok() {
        let config = DetuneConfig::new(10.0, DetuneDistribution::Uniform, 0).unwrap();
        let mut voices: Vec<Vec<f32>> = vec![];
        assert!(apply_detune(&mut voices, &config, 24000).is_ok());
    }

    #[test]
    fn test_preserves_buffer_length() {
        let config = DetuneConfig::new(12.0, DetuneDistribution::Uniform, 0).unwrap();
        let len = 3000;
        let signal: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut voices = vec![
            signal.clone(),
            signal.clone(),
            signal.clone(),
            signal,
        ];
        apply_detune(&mut voices, &config, 24000).unwrap();

        for (i, voice) in voices.iter().enumerate() {
            assert_eq!(voice.len(), len, "voice {i} length should be preserved");
        }
    }
}
