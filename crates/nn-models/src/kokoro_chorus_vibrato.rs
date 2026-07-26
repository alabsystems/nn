// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-voice LFO-based vibrato (F0 pitch modulation) for Kokoro chorus.
//!
//! Real singers have natural vibrato -- a periodic pitch oscillation typically
//! 4-7 Hz at 20-80 cents depth. In a choir, each singer's vibrato has a
//! slightly different rate, depth, and phase, creating a rich, shimmering
//! texture that distinguishes a live ensemble from a unison synth patch.
//!
//! This module applies per-voice vibrato by resampling PCM audio at a
//! time-varying rate controlled by a sinusoidal LFO (low-frequency
//! oscillator). Each voice gets a deterministic phase offset derived from
//! its index, so vibrato patterns differ across voices while remaining
//! reproducible across runs.
//!
//! # Architecture
//!
//! ```text
//! Voice 0: LFO(rate_hz + delta_0, depth_cents + delta_0, phase_0) -> resample
//! Voice 1: LFO(rate_hz + delta_1, depth_cents + delta_1, phase_1) -> resample
//! ...
//! Voice N: LFO(rate_hz + delta_N, depth_cents + delta_N, phase_N) -> resample
//! ```
//!
//! The per-voice deltas spread the LFO parameters symmetrically around the
//! configured center values, creating natural decorrelation between voices.
//!
//! # Placement in the chorus pipeline
//!
//! Vibrato is applied **before** detuning in the processing chain:
//! ```text
//! Per-voice: vibrato -> detuning -> EQ -> de-essing -> humanize
//! ```
//!
//! This ordering is deliberate: vibrato modulates pitch periodically (dynamic),
//! while detuning adds a static pitch offset. Applying vibrato first means
//! the detuning allpass filter operates on already-modulated audio, which is
//! the acoustically correct chain (a singer's vibrato rides on top of their
//! natural pitch offset from the ensemble).
//!
//! # References
//!
//! - Sundberg, J. "The Science of the Singing Voice." Northern Illinois
//!   University Press, 1987. (vibrato rate/depth norms)
//! - d'Alessandro, C. & Doval, B. "Voice quality modification for emotional
//!   speech synthesis." Interspeech, 2003.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for per-voice vibrato (F0 pitch modulation).
///
/// Controls the LFO rate, depth, and per-voice spread. Each voice gets
/// a deterministic variation around the center values based on its index.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VibratoConfig {
    /// Center vibrato rate in Hz (LFO frequency).
    ///
    /// Typical singing vibrato is 4.5-6.5 Hz. Values outside [2.0, 12.0]
    /// sound unnatural but are allowed for creative effects.
    ///
    /// Must be in [0.5, 20.0] and finite. Default: `5.5`.
    pub rate_hz: f32,

    /// Center vibrato depth in cents (peak pitch deviation).
    ///
    /// Typical singing vibrato is 30-80 cents. Subtle chorus vibrato
    /// uses 10-30 cents to avoid obvious pitch wobble.
    ///
    /// Must be in [0.0, 200.0] and finite. Default: `20.0`.
    pub depth_cents: f32,

    /// Per-voice rate spread in Hz.
    ///
    /// Each voice's LFO rate is offset symmetrically around `rate_hz`
    /// by up to `rate_spread_hz`. For example, with rate_hz=5.5 and
    /// rate_spread_hz=0.5, voices get rates in [5.0, 6.0].
    ///
    /// Must be in [0.0, 5.0] and finite. Default: `0.3`.
    pub rate_spread_hz: f32,

    /// Per-voice depth spread in cents.
    ///
    /// Each voice's vibrato depth is offset symmetrically around
    /// `depth_cents` by up to `depth_spread_cents`.
    ///
    /// Must be in [0.0, 100.0] and finite. Default: `5.0`.
    pub depth_spread_cents: f32,

    /// Onset delay in seconds before vibrato reaches full depth.
    ///
    /// Real singers typically start a note straight and add vibrato
    /// after 0.1-0.3 seconds. The vibrato depth ramps linearly from
    /// 0 to `depth_cents` over `onset_sec` seconds.
    ///
    /// Must be in [0.0, 2.0] and finite. Default: `0.15`.
    pub onset_sec: f32,
}

impl Default for VibratoConfig {
    fn default() -> Self {
        Self {
            rate_hz: 5.5,
            depth_cents: 20.0,
            rate_spread_hz: 0.3,
            depth_spread_cents: 5.0,
            onset_sec: 0.15,
        }
    }
}

impl VibratoConfig {
    /// Create a new vibrato configuration with validation.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is non-finite
    /// or outside its valid range.
    pub fn new(
        rate_hz: f32,
        depth_cents: f32,
        rate_spread_hz: f32,
        depth_spread_cents: f32,
        onset_sec: f32,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            rate_hz,
            depth_cents,
            rate_spread_hz,
            depth_spread_cents,
            onset_sec,
        };
        config.validate()?;
        Ok(config)
    }

    /// Create a subtle vibrato preset suitable for background chorus voices.
    ///
    /// Lower depth and rate for a gentle shimmer rather than obvious wobble.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            rate_hz: 5.0,
            depth_cents: 12.0,
            rate_spread_hz: 0.2,
            depth_spread_cents: 3.0,
            onset_sec: 0.2,
        }
    }

    /// Create a natural singing vibrato preset.
    ///
    /// Moderate depth and rate matching typical trained singer vibrato.
    #[must_use]
    pub fn natural() -> Self {
        Self {
            rate_hz: 5.5,
            depth_cents: 40.0,
            rate_spread_hz: 0.4,
            depth_spread_cents: 8.0,
            onset_sec: 0.15,
        }
    }

    /// Validate this configuration.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.rate_hz.is_finite() || !(0.5..=20.0).contains(&self.rate_hz) {
            return Err(KokoroError::InvalidConfig {
                field: "rate_hz",
                reason: format!("must be finite and in [0.5, 20.0], got {}", self.rate_hz),
            });
        }
        if !self.depth_cents.is_finite() || !(0.0..=200.0).contains(&self.depth_cents) {
            return Err(KokoroError::InvalidConfig {
                field: "depth_cents",
                reason: format!(
                    "must be finite and in [0.0, 200.0], got {}",
                    self.depth_cents,
                ),
            });
        }
        if !self.rate_spread_hz.is_finite() || !(0.0..=5.0).contains(&self.rate_spread_hz) {
            return Err(KokoroError::InvalidConfig {
                field: "rate_spread_hz",
                reason: format!(
                    "must be finite and in [0.0, 5.0], got {}",
                    self.rate_spread_hz,
                ),
            });
        }
        if !self.depth_spread_cents.is_finite() || !(0.0..=100.0).contains(&self.depth_spread_cents)
        {
            return Err(KokoroError::InvalidConfig {
                field: "depth_spread_cents",
                reason: format!(
                    "must be finite and in [0.0, 100.0], got {}",
                    self.depth_spread_cents,
                ),
            });
        }
        if !self.onset_sec.is_finite() || !(0.0..=2.0).contains(&self.onset_sec) {
            return Err(KokoroError::InvalidConfig {
                field: "onset_sec",
                reason: format!("must be finite and in [0.0, 2.0], got {}", self.onset_sec),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Per-voice LFO parameters
// ---------------------------------------------------------------------------

/// Compute per-voice LFO parameters (rate, depth, phase) for `n_voices` voices.
///
/// Voice 0 gets the center values with phase 0. Other voices get symmetrically
/// spread rates and depths, with evenly distributed phase offsets to decorrelate
/// the LFO waveforms.
fn voice_lfo_params(config: &VibratoConfig, n_voices: usize) -> Vec<(f32, f32, f32)> {
    let mut params = Vec::with_capacity(n_voices);

    for i in 0..n_voices {
        if i == 0 {
            // Anchor voice: center values, zero phase.
            params.push((config.rate_hz, config.depth_cents, 0.0));
        } else {
            let n_spread = n_voices - 1;
            // t in [0, 1] for voices 1..n_voices.
            let t = if n_spread == 1 {
                0.5
            } else {
                (i - 1) as f64 / (n_spread - 1) as f64
            };

            // Symmetric spread: -spread to +spread.
            let rate_offset = f64::from(config.rate_spread_hz) * (2.0 * t - 1.0);
            let depth_offset = f64::from(config.depth_spread_cents) * (2.0 * t - 1.0);

            let rate = (f64::from(config.rate_hz) + rate_offset).max(0.5) as f32;
            let depth = (f64::from(config.depth_cents) + depth_offset).max(0.0) as f32;

            // Phase offset evenly distributed across [0, 2*pi).
            let phase = (i as f64 / n_voices as f64) * std::f64::consts::TAU;

            params.push((rate, depth, phase as f32));
        }
    }

    params
}

// ---------------------------------------------------------------------------
// Vibrato application
// ---------------------------------------------------------------------------

/// Apply per-voice vibrato to a set of voice audio buffers.
///
/// Each voice is resampled at a time-varying rate controlled by an LFO
/// (low-frequency oscillator) to create periodic pitch modulation. The
/// LFO parameters (rate, depth, phase) differ per voice for natural
/// decorrelation.
///
/// # Arguments
///
/// * `voices` - Mutable slice of per-voice PCM buffers (24kHz mono).
/// * `config` - Vibrato configuration (rate, depth, spread).
/// * `sample_rate` - Audio sample rate in Hz.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid.
pub fn apply_vibrato(
    voices: &mut [Vec<f32>],
    config: &VibratoConfig,
    sample_rate: u32,
) -> Result<(), KokoroError> {
    config.validate()?;

    if voices.is_empty() {
        return Ok(());
    }

    // Skip if vibrato depth is negligible.
    if config.depth_cents < 0.1 {
        return Ok(());
    }

    let params = voice_lfo_params(config, voices.len());
    let sr = f64::from(sample_rate);
    let onset_samples = (f64::from(config.onset_sec) * sr) as usize;

    for (voice_idx, voice_pcm) in voices.iter_mut().enumerate() {
        let (rate, depth, phase) = params[voice_idx];

        // Skip voices with negligible depth.
        if depth < 0.1 {
            continue;
        }

        let len = voice_pcm.len();
        if len == 0 {
            continue;
        }

        let mut resampled = Vec::with_capacity(len);

        // For each output sample, compute the instantaneous source position.
        // The LFO modulates the pitch by `depth` cents, which corresponds
        // to a time-varying playback rate of 2^(cents * sin(lfo) / 1200).
        //
        // For small depths (<100 cents), the approximation
        //   rate - 1 ~ depth * ln(2)/1200 * sin(2*pi*f*t + phi)
        // is accurate, and its integral is:
        //   offset(t) = -depth * ln(2) / (1200 * 2*pi*f) * cos(2*pi*f*t + phi)
        //
        // We use this closed-form to avoid cumulative drift from numerical
        // integration.
        let rate_f64 = f64::from(rate);
        let depth_f64 = f64::from(depth);
        let phase_f64 = f64::from(phase);
        let ln2 = std::f64::consts::LN_2;
        let tau = std::f64::consts::TAU;

        // Max sample offset amplitude (in samples).
        let max_offset_samples = depth_f64 * ln2 / (1200.0 * tau * rate_f64) * sr;

        for i in 0..len {
            // Onset ramp: linearly ramp vibrato depth from 0 to 1 over onset_samples.
            let onset_gain = if onset_samples > 0 && i < onset_samples {
                i as f64 / onset_samples as f64
            } else {
                1.0
            };

            let t = i as f64 / sr;
            let lfo_cos = (tau * rate_f64 * t + phase_f64).cos();

            // Cumulative sample offset from vibrato.
            let offset = -onset_gain * max_offset_samples * lfo_cos;

            let src_pos = i as f64 + offset;
            let src_idx = src_pos.floor() as isize;
            let frac = (src_pos - src_idx as f64) as f32;

            // Linear interpolation with bounds clamping.
            let s0 = if src_idx >= 0 && (src_idx as usize) < len {
                voice_pcm[src_idx as usize]
            } else if src_idx < 0 {
                voice_pcm[0]
            } else {
                voice_pcm[len - 1]
            };

            let s1_idx = src_idx + 1;
            let s1 = if s1_idx >= 0 && (s1_idx as usize) < len {
                voice_pcm[s1_idx as usize]
            } else if s1_idx < 0 {
                voice_pcm[0]
            } else {
                voice_pcm[len - 1]
            };

            let sample = s0 + frac * (s1 - s0);
            // Guard against NaN/Inf from edge cases.
            resampled.push(if sample.is_finite() { sample } else { 0.0 });
        }

        // Replace original buffer.
        voice_pcm.clear();
        voice_pcm.extend_from_slice(&resampled);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_depth_is_identity() {
        let config = VibratoConfig::new(5.5, 0.0, 0.0, 0.0, 0.0).unwrap();
        let original: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut voices = vec![original.clone(), original.clone(), original.clone()];
        apply_vibrato(&mut voices, &config, 24000).unwrap();

        for (vi, voice) in voices.iter().enumerate() {
            for (j, (&got, &expected)) in voice.iter().zip(original.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-5,
                    "voice {vi} sample {j}: got {got}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn test_vibrato_changes_audio() {
        let config = VibratoConfig::default();
        let signal: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal.clone(), signal.clone(), signal.clone()];
        apply_vibrato(&mut voices, &config, 24000).unwrap();

        // Voices should differ from original (except possibly during onset).
        let skip = 4000; // skip onset period
        let diff: f32 = voices[1][skip..]
            .iter()
            .zip(signal[skip..].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (voices[1].len() - skip) as f32;

        assert!(
            diff > 1e-5,
            "vibrato should modify the audio, mean diff = {diff}",
        );
    }

    #[test]
    fn test_preserves_buffer_length() {
        let config = VibratoConfig::default();
        let len = 6000;
        let signal: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut voices = vec![
            signal.clone(),
            signal.clone(),
            signal.clone(),
            signal,
        ];
        apply_vibrato(&mut voices, &config, 24000).unwrap();

        for (i, voice) in voices.iter().enumerate() {
            assert_eq!(voice.len(), len, "voice {i} length should be preserved");
        }
    }

    #[test]
    fn test_voices_differ_from_each_other() {
        let config = VibratoConfig::default();
        let signal: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![
            signal.clone(),
            signal.clone(),
            signal.clone(),
            signal,
        ];
        apply_vibrato(&mut voices, &config, 24000).unwrap();

        // Voice 1 and voice 2 should differ (different LFO params).
        let skip = 4000;
        let diff: f32 = voices[1][skip..]
            .iter()
            .zip(voices[2][skip..].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (voices[1].len() - skip) as f32;

        assert!(
            diff > 1e-6,
            "different voices should have different vibrato, mean diff = {diff}",
        );
    }

    #[test]
    fn test_onset_ramp() {
        let config = VibratoConfig::new(5.5, 50.0, 0.0, 0.0, 0.5).unwrap();
        let signal: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal.clone()];
        apply_vibrato(&mut voices, &config, 24000).unwrap();

        // Early samples (within onset ramp) should be closer to original
        // than later samples (full vibrato).
        let early_diff: f32 = voices[0][..1000]
            .iter()
            .zip(signal[..1000].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 1000.0;

        let late_diff: f32 = voices[0][18000..19000]
            .iter()
            .zip(signal[18000..19000].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 1000.0;

        assert!(
            early_diff < late_diff + 1e-3,
            "onset should have less vibrato: early={early_diff}, late={late_diff}",
        );
    }

    #[test]
    fn test_config_validation_rejects_invalid() {
        assert!(VibratoConfig::new(0.0, 20.0, 0.0, 0.0, 0.0).is_err()); // rate too low
        assert!(VibratoConfig::new(25.0, 20.0, 0.0, 0.0, 0.0).is_err()); // rate too high
        assert!(VibratoConfig::new(5.5, -1.0, 0.0, 0.0, 0.0).is_err()); // negative depth
        assert!(VibratoConfig::new(5.5, 201.0, 0.0, 0.0, 0.0).is_err()); // depth too high
        assert!(VibratoConfig::new(5.5, 20.0, 6.0, 0.0, 0.0).is_err()); // rate_spread too high
        assert!(VibratoConfig::new(5.5, 20.0, 0.0, 101.0, 0.0).is_err()); // depth_spread too high
        assert!(VibratoConfig::new(5.5, 20.0, 0.0, 0.0, 3.0).is_err()); // onset too long
        assert!(VibratoConfig::new(f32::NAN, 20.0, 0.0, 0.0, 0.0).is_err());
        assert!(VibratoConfig::new(5.5, f32::INFINITY, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_config_validation_accepts_valid() {
        assert!(VibratoConfig::new(0.5, 0.0, 0.0, 0.0, 0.0).is_ok());
        assert!(VibratoConfig::new(20.0, 200.0, 5.0, 100.0, 2.0).is_ok());
        assert!(VibratoConfig::new(5.5, 20.0, 0.3, 5.0, 0.15).is_ok());
    }

    #[test]
    fn test_empty_voices_ok() {
        let config = VibratoConfig::default();
        let mut voices: Vec<Vec<f32>> = vec![];
        assert!(apply_vibrato(&mut voices, &config, 24000).is_ok());
    }

    #[test]
    fn test_no_nan_in_output() {
        let config = VibratoConfig::new(5.5, 80.0, 1.0, 20.0, 0.1).unwrap();
        let signal: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal.clone(), signal.clone(), signal];
        apply_vibrato(&mut voices, &config, 24000).unwrap();

        for (vi, voice) in voices.iter().enumerate() {
            for (j, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "voice {vi} sample {j} is non-finite: {s}");
            }
        }
    }

    #[test]
    fn test_presets() {
        let subtle = VibratoConfig::subtle();
        assert!(subtle.validate().is_ok());
        assert!(subtle.depth_cents < 20.0);

        let natural = VibratoConfig::natural();
        assert!(natural.validate().is_ok());
        assert!(natural.depth_cents > 30.0);
    }

    #[test]
    fn test_voice_lfo_params_spread() {
        let config = VibratoConfig::new(5.5, 20.0, 1.0, 10.0, 0.0).unwrap();
        let params = voice_lfo_params(&config, 5);

        // Voice 0: center values, zero phase.
        assert!((params[0].0 - 5.5).abs() < 1e-6, "voice 0 rate");
        assert!((params[0].1 - 20.0).abs() < 1e-6, "voice 0 depth");
        assert!((params[0].2).abs() < 1e-6, "voice 0 phase");

        // Other voices should have different phases.
        for i in 1..5 {
            assert!(
                params[i].2.abs() > 1e-6,
                "voice {i} should have non-zero phase",
            );
        }

        // Rates should be spread around 5.5.
        let rates: Vec<f32> = params.iter().map(|p| p.0).collect();
        let min_rate = rates.iter().copied().fold(f32::INFINITY, f32::min);
        let max_rate = rates.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max_rate - min_rate) > 0.5,
            "rates should be spread: min={min_rate}, max={max_rate}",
        );
    }
}
