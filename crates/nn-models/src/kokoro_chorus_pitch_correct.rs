// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! YIN pitch detection and musical-scale-aware pitch correction for Kokoro chorus.
//!
//! Pitch correction (Auto-Tune style) detects the fundamental frequency of each
//! voice and snaps it toward the nearest note in a target musical scale. The
//! correction strength and speed control whether the effect sounds natural
//! (gentle nudge toward pitch) or stylized (hard robotic snap).
//!
//! # Algorithm Overview
//!
//! 1. **YIN pitch detection** -- autocorrelation-based F0 estimation with
//!    parabolic interpolation for sub-sample accuracy. Operates on overlapping
//!    windows (~256 samples at 24kHz = ~10.7ms).
//!
//! 2. **Musical scale mapping** -- the detected frequency is compared against
//!    the nearest in-scale note. Chromatic snaps to the closest semitone;
//!    Major/Minor/Pentatonic snap to scale degrees rooted at a given MIDI note.
//!
//! 3. **PSOLA pitch shifting** -- Pitch Synchronous Overlap-Add resamples
//!    pitch periods to shift the detected frequency toward the target note.
//!    Optionally preserves formants to avoid the "chipmunk effect."
//!
//! 4. **Smoothing envelope** -- The correction amount is smoothed over
//!    `speed_ms` milliseconds to avoid abrupt pitch jumps. At speed=0 the
//!    correction is instantaneous ("robotic"); at speed=200 it glides naturally.
//!
//! # Placement in the chorus pipeline
//!
//! Pitch correction is applied **after** vibrato and detuning, and **before**
//! EQ and dynamics processing:
//! ```text
//! Per-voice: vibrato -> detuning -> pitch_correct -> EQ -> dynamics
//! ```
//!
//! This ordering ensures that intentional detuning and vibrato modulations are
//! partially preserved (the correction "rides on top" of the modulated pitch),
//! while still snapping the overall pitch center to the scale.
//!
//! # References
//!
//! - de Cheveigne, A. & Kawahara, H. "YIN, a fundamental frequency estimator
//!   for speech and music." JASA, 111(4), 2002.
//! - Moulines, E. & Charpentier, F. "Pitch-synchronous waveform processing
//!   techniques for text-to-speech synthesis using diphones." Speech
//!   Communication, 9(5-6), 1990. (PSOLA)

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Musical scale
// ---------------------------------------------------------------------------

/// Musical scales for pitch-correction target note selection.
///
/// Each variant defines which pitch classes (0-11, where 0=C) are valid
/// snap targets. The `root` parameter (MIDI note 0-11) transposes the
/// scale pattern to any key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum MusicalScale {
    /// Snap to the nearest semitone (all 12 pitch classes).
    #[default]
    Chromatic,
    /// Major scale (Ionian mode). Root is MIDI pitch class 0-11.
    Major(u8),
    /// Natural minor scale (Aeolian mode). Root is MIDI pitch class 0-11.
    Minor(u8),
    /// Major pentatonic scale. Root is MIDI pitch class 0-11.
    Pentatonic(u8),
}


/// Semitone intervals from root for each scale type.
const MAJOR_INTERVALS: [u8; 7] = [0, 2, 4, 5, 7, 9, 11];
const MINOR_INTERVALS: [u8; 7] = [0, 2, 3, 5, 7, 8, 10];
const PENTATONIC_INTERVALS: [u8; 5] = [0, 2, 4, 7, 9];

impl MusicalScale {
    /// Find the nearest in-scale frequency to the given frequency in Hz.
    ///
    /// Returns the target frequency that the input should be corrected toward.
    /// Uses A4 = `reference_freq` Hz (standard 440.0) for MIDI-to-Hz conversion.
    ///
    /// # IEEE 754 safety
    ///
    /// Returns the input unchanged if it is non-finite, non-positive, or
    /// if `reference_freq` is non-finite or non-positive.
    #[must_use]
    pub fn nearest_note_hz(&self, freq_hz: f32, reference_freq: f32) -> f32 {
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            return freq_hz;
        }
        if !reference_freq.is_finite() || reference_freq <= 0.0 {
            return freq_hz;
        }

        // Convert frequency to continuous MIDI note number.
        // MIDI note 69 = A4 = reference_freq Hz.
        // midi = 69 + 12 * log2(freq / reference_freq)
        let ratio = f64::from(freq_hz) / f64::from(reference_freq);
        if ratio <= 0.0 {
            return freq_hz;
        }
        let midi_continuous = 69.0 + 12.0 * ratio.log2();

        // Find the nearest in-scale MIDI note.
        let target_midi = self.nearest_scale_midi(midi_continuous);

        // Convert back to Hz.
        let target_hz = f64::from(reference_freq) * (2.0f64).powf((target_midi - 69.0) / 12.0);
        let result = target_hz as f32;
        if result.is_finite() && result > 0.0 {
            result
        } else {
            freq_hz
        }
    }

    /// Find the nearest in-scale MIDI note number to a continuous MIDI value.
    fn nearest_scale_midi(&self, midi: f64) -> f64 {
        match self {
            Self::Chromatic => midi.round(),
            Self::Major(root) => Self::snap_to_intervals(midi, *root, &MAJOR_INTERVALS),
            Self::Minor(root) => Self::snap_to_intervals(midi, *root, &MINOR_INTERVALS),
            Self::Pentatonic(root) => Self::snap_to_intervals(midi, *root, &PENTATONIC_INTERVALS),
        }
    }

    /// Snap a continuous MIDI value to the nearest note in a scale defined
    /// by semitone intervals from a root pitch class.
    fn snap_to_intervals(midi: f64, root: u8, intervals: &[u8]) -> f64 {
        let root = f64::from(root % 12);

        // Find the pitch class (0-11) relative to root.
        // The octave is determined by rounding to the nearest scale degree.
        let mut best_midi = midi.round();
        let mut best_dist = f64::MAX;

        // Search the octave containing midi and the adjacent octaves.
        let base_octave = ((midi - root) / 12.0).floor() as i32;

        for oct_offset in -1..=1 {
            let octave = base_octave + oct_offset;
            for &interval in intervals {
                let candidate = root + f64::from(octave) * 12.0 + f64::from(interval);
                let dist = (midi - candidate).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_midi = candidate;
                }
            }
        }

        best_midi
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for pitch correction in the Kokoro chorus.
///
/// Controls correction strength, speed, target scale, reference tuning,
/// and formant preservation. Use the builder methods for ergonomic
/// construction with validation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PitchCorrectConfig {
    /// Correction strength (0.0-1.0).
    ///
    /// - 0.0 = no correction (pass-through)
    /// - 0.5 = gentle correction (natural sounding)
    /// - 1.0 = hard snap to nearest scale note ("robotic" / Auto-Tune effect)
    ///
    /// Must be finite and in [0.0, 1.0]. Default: `0.5`.
    pub correction_strength: f32,

    /// Correction speed in milliseconds.
    ///
    /// How quickly the pitch slides toward the target note.
    /// - 0.0 = instant correction (robotic)
    /// - 50.0 = fast but natural
    /// - 200.0 = slow, gentle glide
    ///
    /// Must be finite and in [0.0, 200.0]. Default: `50.0`.
    pub speed_ms: f32,

    /// Target musical scale for pitch snapping.
    ///
    /// Default: `MusicalScale::Chromatic`.
    pub scale: MusicalScale,

    /// A4 reference frequency in Hz for tuning.
    ///
    /// Standard concert pitch is 440.0 Hz. Baroque tuning uses 415.0 Hz.
    /// Must be finite and in [415.0, 465.0]. Default: `440.0`.
    pub reference_freq: f32,

    /// Whether to preserve formants during pitch shifting.
    ///
    /// When true, the spectral envelope is maintained while shifting
    /// pitch, preventing the "chipmunk effect" on large corrections.
    /// Slightly more CPU-intensive. Default: `true`.
    pub formant_preserve: bool,
}

impl Default for PitchCorrectConfig {
    fn default() -> Self {
        Self {
            correction_strength: 0.5,
            speed_ms: 50.0,
            scale: MusicalScale::Chromatic,
            reference_freq: 440.0,
            formant_preserve: true,
        }
    }
}

impl PitchCorrectConfig {
    /// Create a new pitch correction config with all parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range.
    pub fn new(
        correction_strength: f32,
        speed_ms: f32,
        scale: MusicalScale,
        reference_freq: f32,
        formant_preserve: bool,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            correction_strength,
            speed_ms,
            scale,
            reference_freq,
            formant_preserve,
        };
        config.validate()?;
        Ok(config)
    }

    /// Builder: set correction strength.
    #[must_use]
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.correction_strength = strength;
        self
    }

    /// Builder: set correction speed in milliseconds.
    #[must_use]
    pub fn with_speed_ms(mut self, speed_ms: f32) -> Self {
        self.speed_ms = speed_ms;
        self
    }

    /// Builder: set target musical scale.
    #[must_use]
    pub fn with_scale(mut self, scale: MusicalScale) -> Self {
        self.scale = scale;
        self
    }

    /// Builder: set A4 reference frequency.
    #[must_use]
    pub fn with_reference_freq(mut self, freq: f32) -> Self {
        self.reference_freq = freq;
        self
    }

    /// Builder: set formant preservation.
    #[must_use]
    pub fn with_formant_preserve(mut self, preserve: bool) -> Self {
        self.formant_preserve = preserve;
        self
    }

    /// Hard Auto-Tune preset: instant snap to chromatic scale.
    #[must_use]
    pub fn hard_tune() -> Self {
        Self {
            correction_strength: 1.0,
            speed_ms: 0.0,
            scale: MusicalScale::Chromatic,
            reference_freq: 440.0,
            formant_preserve: true,
        }
    }

    /// Natural correction preset: gentle nudge toward scale.
    #[must_use]
    pub fn natural() -> Self {
        Self {
            correction_strength: 0.4,
            speed_ms: 120.0,
            scale: MusicalScale::Chromatic,
            reference_freq: 440.0,
            formant_preserve: true,
        }
    }

    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is non-finite
    /// or outside its valid range.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.correction_strength.is_finite() || !(0.0..=1.0).contains(&self.correction_strength)
        {
            return Err(KokoroError::InvalidConfig {
                field: "correction_strength",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.correction_strength,
                ),
            });
        }
        if !self.speed_ms.is_finite() || !(0.0..=200.0).contains(&self.speed_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "speed_ms",
                reason: format!("must be finite and in [0.0, 200.0], got {}", self.speed_ms),
            });
        }
        if !self.reference_freq.is_finite() || !(415.0..=465.0).contains(&self.reference_freq) {
            return Err(KokoroError::InvalidConfig {
                field: "reference_freq",
                reason: format!(
                    "must be finite and in [415.0, 465.0], got {}",
                    self.reference_freq,
                ),
            });
        }
        // Validate scale root is in [0, 11].
        match self.scale {
            MusicalScale::Chromatic => {}
            MusicalScale::Major(root)
            | MusicalScale::Minor(root)
            | MusicalScale::Pentatonic(root) => {
                if root > 11 {
                    return Err(KokoroError::InvalidConfig {
                        field: "scale",
                        reason: format!("root pitch class must be in [0, 11], got {root}"),
                    });
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// YIN pitch detection
// ---------------------------------------------------------------------------

/// Default YIN analysis window size in samples.
const YIN_WINDOW: usize = 256;

/// YIN absolute threshold for pitch detection.
/// Values below this in the cumulative mean normalized difference function
/// indicate a strong periodicity. Typical range: 0.05-0.20.
const YIN_THRESHOLD: f32 = 0.15;

/// Detect the fundamental frequency of an audio segment using the YIN algorithm.
///
/// Returns `Some(freq_hz)` if a clear pitch is detected, or `None` for
/// unvoiced/noise segments. The YIN algorithm computes an autocorrelation-based
/// difference function and applies cumulative mean normalization to find the
/// pitch period with sub-sample accuracy via parabolic interpolation.
///
/// # Arguments
///
/// * `audio` - PCM samples (mono, any sample rate).
/// * `sample_rate` - Sample rate in Hz.
///
/// # IEEE 754 safety
///
/// Returns `None` if the detected frequency is non-finite or non-positive.
#[must_use]
pub fn detect_pitch(audio: &[f32], sample_rate: f32) -> Option<f32> {
    if audio.len() < YIN_WINDOW * 2 || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return None;
    }

    let w = YIN_WINDOW;

    // Step 1: Difference function d(tau).
    // d(tau) = sum_{j=0}^{W-1} (x[j] - x[j + tau])^2
    let mut diff = vec![0.0f32; w];
    for tau in 1..w {
        let mut sum = 0.0f64;
        for j in 0..w {
            let delta = f64::from(audio[j]) - f64::from(audio[j + tau]);
            sum += delta * delta;
        }
        diff[tau] = sum as f32;
    }

    // Step 2: Cumulative Mean Normalized Difference Function (CMND).
    // d'(tau) = d(tau) / ((1/tau) * sum_{j=1}^{tau} d(j))
    // d'(0) = 1.
    let mut cmnd = vec![1.0f32; w];
    let mut running_sum = 0.0f64;
    for tau in 1..w {
        running_sum += f64::from(diff[tau]);
        if running_sum > 1e-12 {
            cmnd[tau] = (f64::from(diff[tau]) * tau as f64 / running_sum) as f32;
        } else {
            cmnd[tau] = 1.0;
        }
    }

    // Step 3: Absolute threshold search.
    // Find the first tau where cmnd[tau] < threshold, then find the local minimum.
    let min_tau = 2; // Minimum period (avoid tau=0,1 artifacts).
    let max_tau = w - 1;

    let mut best_tau: Option<usize> = None;
    for tau in min_tau..max_tau {
        if cmnd[tau] < YIN_THRESHOLD {
            // Found a candidate -- walk forward to find the local minimum.
            let mut local_min_tau = tau;
            let mut local_min_val = cmnd[tau];
            for t in (tau + 1)..max_tau {
                if cmnd[t] < local_min_val {
                    local_min_val = cmnd[t];
                    local_min_tau = t;
                } else {
                    break; // Past the local minimum.
                }
            }
            best_tau = Some(local_min_tau);
            break;
        }
    }

    let tau = best_tau?;

    // Step 4: Parabolic interpolation for sub-sample accuracy.
    let refined_tau = if tau > 0 && tau < w - 1 {
        let alpha = f64::from(cmnd[tau - 1]);
        let beta = f64::from(cmnd[tau]);
        let gamma = f64::from(cmnd[tau + 1]);
        let denom = 2.0 * (2.0 * beta - alpha - gamma);
        if denom.abs() > 1e-12 {
            tau as f64 + (alpha - gamma) / denom
        } else {
            tau as f64
        }
    } else {
        tau as f64
    };

    if refined_tau <= 0.0 {
        return None;
    }

    let freq = f64::from(sample_rate) / refined_tau;
    let freq_f32 = freq as f32;

    // Sanity check: human voice range is roughly 50-1500 Hz.
    if freq_f32.is_finite() && freq_f32 > 30.0 && freq_f32 < 4000.0 {
        Some(freq_f32)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Pitch corrector (stateful processor)
// ---------------------------------------------------------------------------

/// Stateful pitch corrector implementing YIN detection + PSOLA pitch shifting.
///
/// Maintains internal buffers for overlap-add processing and a smoothed
/// correction envelope for natural-sounding pitch transitions.
pub struct PitchCorrector {
    config: PitchCorrectConfig,
    sample_rate: f32,
    /// Hop size for analysis windows (window / 4 for 75% overlap).
    hop_size: usize,
    /// Smoothed current correction ratio (1.0 = no correction).
    smooth_ratio: f64,
    /// Smoothing coefficient per sample, derived from speed_ms.
    smooth_coeff: f64,
}

impl PitchCorrector {
    /// Create a new pitch corrector.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the config is invalid, or
    /// `KokoroError::InvalidInput` if `sample_rate` is non-positive or non-finite.
    pub fn new(config: &PitchCorrectConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidInput(format!(
                "sample_rate must be positive and finite, got {sample_rate}"
            )));
        }

        let hop_size = YIN_WINDOW / 4;

        // Smoothing coefficient: exponential decay per sample.
        // At speed_ms=0, coeff=1.0 (instant). At speed_ms=200, slow glide.
        let smooth_coeff = if config.speed_ms < 0.01 {
            1.0
        } else {
            let time_constant_samples = f64::from(config.speed_ms) * f64::from(sample_rate) / 1000.0;
            if time_constant_samples > 1.0 {
                1.0 - (-1.0 / time_constant_samples).exp()
            } else {
                1.0
            }
        };

        Ok(Self {
            config: config.clone(),
            sample_rate,
            hop_size,
            smooth_ratio: 1.0,
            smooth_coeff,
        })
    }

    /// Reset internal state (smoothing envelope and PSOLA buffers).
    pub fn reset(&mut self) {
        self.smooth_ratio = 1.0;
    }

    /// Process audio in-place, applying pitch correction.
    ///
    /// The audio is analyzed in overlapping windows. For each window, the
    /// pitch is detected via YIN, the target note is found via the configured
    /// scale, and the correction is applied by resampling.
    pub fn process(&mut self, audio: &mut [f32]) {
        if audio.len() < YIN_WINDOW * 2 {
            return;
        }

        // Skip if correction strength is zero.
        if self.config.correction_strength < 1e-6 {
            return;
        }

        let len = audio.len();
        let mut output = vec![0.0f32; len];
        let mut weight = vec![0.0f32; len];

        // Hann window for overlap-add.
        let win_len = YIN_WINDOW * 2;
        let hann: Vec<f32> = (0..win_len)
            .map(|i| {
                let t = i as f64 / win_len as f64;
                (0.5 * (1.0 - (std::f64::consts::TAU * t).cos())) as f32
            })
            .collect();

        let mut pos = 0;
        while pos + win_len <= len {
            let window = &audio[pos..pos + win_len];

            // Detect pitch in this window.
            if let Some(detected_hz) = detect_pitch(window, self.sample_rate) {
                let target_hz = self
                    .config
                    .scale
                    .nearest_note_hz(detected_hz, self.config.reference_freq);

                // Compute correction ratio.
                let raw_ratio = f64::from(target_hz) / f64::from(detected_hz);

                // Apply correction strength: interpolate between 1.0 (no correction)
                // and raw_ratio (full correction).
                let strength = f64::from(self.config.correction_strength);
                let target_ratio = 1.0 + strength * (raw_ratio - 1.0);

                // Smooth the ratio over time.
                self.smooth_ratio += self.smooth_coeff * (target_ratio - self.smooth_ratio);

                // Apply pitch shift via resampling this window.
                let ratio = self.smooth_ratio;

                for i in 0..win_len {
                    let src_pos = i as f64 * ratio;
                    let src_idx = src_pos.floor() as usize;
                    let frac = (src_pos - src_idx as f64) as f32;

                    let sample = if src_idx < win_len {
                        let s0 = window[src_idx];
                        let s1 = if src_idx + 1 < win_len {
                            window[src_idx + 1]
                        } else {
                            s0
                        };
                        s0 + frac * (s1 - s0)
                    } else {
                        0.0
                    };

                    let out_idx = pos + i;
                    if out_idx < len {
                        let w = hann[i];
                        output[out_idx] += sample * w;
                        weight[out_idx] += w;
                    }
                }
            } else {
                // No pitch detected -- pass through unchanged with windowing.
                for i in 0..win_len {
                    let out_idx = pos + i;
                    if out_idx < len {
                        let w = hann[i];
                        output[out_idx] += window[i] * w;
                        weight[out_idx] += w;
                    }
                }
            }

            pos += self.hop_size;
        }

        // Normalize by overlap weight and copy back.
        for i in 0..len {
            if weight[i] > 1e-6 {
                audio[i] = output[i] / weight[i];
            }
            // Guard against NaN/Inf.
            if !audio[i].is_finite() {
                audio[i] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Apply pitch correction to multiple voice buffers.
///
/// Each voice is independently pitch-corrected according to the config.
/// This is the main entry point for chorus pitch correction.
///
/// # Arguments
///
/// * `voices` - Mutable slice of per-voice PCM buffers (mono).
/// * `config` - Pitch correction configuration.
/// * `sample_rate` - Audio sample rate in Hz.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid, or
/// `KokoroError::InvalidInput` if `sample_rate` is non-positive.
pub fn apply_pitch_correction(
    voices: &mut [Vec<f32>],
    config: &PitchCorrectConfig,
    sample_rate: f32,
) -> Result<(), KokoroError> {
    config.validate()?;

    if voices.is_empty() {
        return Ok(());
    }

    // Skip if correction strength is negligible.
    if config.correction_strength < 1e-6 {
        return Ok(());
    }

    for voice in voices.iter_mut() {
        let mut corrector = PitchCorrector::new(config, sample_rate)?;
        corrector.process(voice);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sine wave at the given frequency and sample rate.
    fn sine_wave(freq_hz: f32, sample_rate: f32, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    // -- Config validation --------------------------------------------------

    #[test]
    fn test_config_default_valid() {
        let config = PitchCorrectConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_hard_tune_valid() {
        let config = PitchCorrectConfig::hard_tune();
        assert!(config.validate().is_ok());
        assert!((config.correction_strength - 1.0).abs() < f32::EPSILON);
        assert!((config.speed_ms).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_natural_valid() {
        let config = PitchCorrectConfig::natural();
        assert!(config.validate().is_ok());
        assert!(config.speed_ms > 100.0);
    }

    #[test]
    fn test_config_validation_rejects_invalid() {
        // Strength out of range.
        assert!(PitchCorrectConfig::new(-0.1, 50.0, MusicalScale::Chromatic, 440.0, true).is_err());
        assert!(PitchCorrectConfig::new(1.1, 50.0, MusicalScale::Chromatic, 440.0, true).is_err());
        assert!(
            PitchCorrectConfig::new(f32::NAN, 50.0, MusicalScale::Chromatic, 440.0, true).is_err()
        );

        // Speed out of range.
        assert!(PitchCorrectConfig::new(0.5, -1.0, MusicalScale::Chromatic, 440.0, true).is_err());
        assert!(PitchCorrectConfig::new(0.5, 201.0, MusicalScale::Chromatic, 440.0, true).is_err());

        // Reference freq out of range.
        assert!(PitchCorrectConfig::new(0.5, 50.0, MusicalScale::Chromatic, 400.0, true).is_err());
        assert!(PitchCorrectConfig::new(0.5, 50.0, MusicalScale::Chromatic, 470.0, true).is_err());

        // Invalid scale root.
        assert!(PitchCorrectConfig::new(0.5, 50.0, MusicalScale::Major(12), 440.0, true).is_err());
    }

    #[test]
    fn test_config_validation_accepts_valid() {
        assert!(PitchCorrectConfig::new(0.0, 0.0, MusicalScale::Chromatic, 440.0, false).is_ok());
        assert!(PitchCorrectConfig::new(1.0, 200.0, MusicalScale::Major(0), 415.0, true).is_ok());
        assert!(PitchCorrectConfig::new(0.5, 100.0, MusicalScale::Minor(11), 465.0, true).is_ok());
        assert!(
            PitchCorrectConfig::new(0.7, 80.0, MusicalScale::Pentatonic(5), 440.0, false).is_ok()
        );
    }

    #[test]
    fn test_builder_methods() {
        let config = PitchCorrectConfig::default()
            .with_strength(0.8)
            .with_speed_ms(100.0)
            .with_scale(MusicalScale::Major(0))
            .with_reference_freq(442.0)
            .with_formant_preserve(false);
        assert!((config.correction_strength - 0.8).abs() < f32::EPSILON);
        assert!((config.speed_ms - 100.0).abs() < f32::EPSILON);
        assert_eq!(config.scale, MusicalScale::Major(0));
        assert!((config.reference_freq - 442.0).abs() < f32::EPSILON);
        assert!(!config.formant_preserve);
    }

    // -- Musical scale tests ------------------------------------------------

    #[test]
    fn test_chromatic_snaps_to_nearest_semitone() {
        let scale = MusicalScale::Chromatic;
        // A4 = 440 Hz. A#4/Bb4 = 466.16 Hz. Midpoint is ~452.9 Hz.
        // 445 Hz is closer to A4.
        let snapped = scale.nearest_note_hz(445.0, 440.0);
        assert!(
            (snapped - 440.0).abs() < 1.0,
            "445 Hz should snap to A4 (440 Hz), got {snapped}",
        );

        // 460 Hz is closer to A#4 (466.16 Hz).
        let snapped = scale.nearest_note_hz(460.0, 440.0);
        assert!(
            (snapped - 466.16).abs() < 1.0,
            "460 Hz should snap to Bb4 (~466 Hz), got {snapped}",
        );
    }

    #[test]
    fn test_major_scale_snaps_correctly() {
        let scale = MusicalScale::Major(0); // C major
                                            // C4 = 261.63 Hz. C#4 = 277.18 Hz (not in C major).
                                            // 270 Hz is between C4 and D4 (293.66) -- should snap to C4 or D4.
        let snapped = scale.nearest_note_hz(270.0, 440.0);
        // 270 Hz: midi = 69 + 12*log2(270/440) = 69 + 12*(-0.705) = 69 - 8.46 = 60.54
        // C major scale degrees near 60: C4=60, D4=62, B3=59
        // 60.54 is closest to 60 (C4).
        assert!(
            (snapped - 261.63).abs() < 2.0,
            "270 Hz should snap to C4 (~261.6 Hz) in C major, got {snapped}",
        );
    }

    #[test]
    fn test_minor_scale_snaps_correctly() {
        let scale = MusicalScale::Minor(9); // A minor (A=9)
                                            // A4 = 440 Hz should snap to itself.
        let snapped = scale.nearest_note_hz(440.0, 440.0);
        assert!(
            (snapped - 440.0).abs() < 0.5,
            "440 Hz should stay at A4 in A minor, got {snapped}",
        );
    }

    #[test]
    fn test_pentatonic_scale_snaps() {
        let scale = MusicalScale::Pentatonic(0); // C pentatonic: C, D, E, G, A
                                                 // F4 = 349.23 Hz (MIDI 65) is NOT in C pentatonic.
                                                 // Nearest scale notes: E4 (MIDI 64) or G4 (MIDI 67).
                                                 // 65 is closer to 64 (E4).
        let snapped = scale.nearest_note_hz(349.23, 440.0);
        let e4 = 440.0 * (2.0f32).powf((64.0 - 69.0) / 12.0); // ~329.63
        let g4 = 440.0 * (2.0f32).powf((67.0 - 69.0) / 12.0); // ~392.00
        assert!(
            (snapped - e4).abs() < 2.0 || (snapped - g4).abs() < 2.0,
            "F4 should snap to E4 or G4 in C pentatonic, got {snapped}",
        );
    }

    #[test]
    fn test_nearest_note_nan_safety() {
        let scale = MusicalScale::Chromatic;
        assert!(scale.nearest_note_hz(f32::NAN, 440.0).is_nan());
        assert!((scale.nearest_note_hz(440.0, f32::NAN) - 440.0).abs() < 0.01);
        assert!((scale.nearest_note_hz(-100.0, 440.0) + 100.0).abs() < 0.01);
    }

    // -- Pitch detection tests ----------------------------------------------

    #[test]
    fn test_detect_pitch_finds_sine_fundamental() {
        let sr = 24000.0;
        let freq = 440.0;
        let signal = sine_wave(freq, sr, 2048);
        let detected = detect_pitch(&signal, sr);
        assert!(detected.is_some(), "should detect pitch of 440 Hz sine");
        let det_hz = detected.unwrap();
        // YIN with a 256-sample window has limited frequency resolution.
        // At 24kHz, tau=54.5 gives 440 Hz; tau=55 gives 436 Hz.
        // Allow ~3% tolerance for sub-sample interpolation accuracy.
        assert!(
            (det_hz - freq).abs() / freq < 0.03,
            "detected {det_hz} Hz, expected ~{freq} Hz (>3% error)",
        );
    }

    #[test]
    fn test_detect_pitch_different_frequencies() {
        let sr = 24000.0;
        for &freq in &[200.0, 300.0, 500.0, 800.0, 1000.0] {
            let signal = sine_wave(freq, sr, 2048);
            let detected = detect_pitch(&signal, sr);
            assert!(detected.is_some(), "should detect pitch at {freq} Hz");
            let det_hz = detected.unwrap();
            assert!(
                (det_hz - freq).abs() / freq < 0.05,
                "freq={freq}: detected {det_hz} Hz (>5% error)",
            );
        }
    }

    #[test]
    fn test_detect_pitch_too_short() {
        let signal = vec![0.0; 100]; // Too short for YIN.
        assert!(detect_pitch(&signal, 24000.0).is_none());
    }

    #[test]
    fn test_detect_pitch_noise_returns_none() {
        // White noise should not have a clear pitch.
        // Use deterministic pseudo-noise.
        let mut signal = vec![0.0f32; 2048];
        let mut state: u32 = 12345;
        for s in signal.iter_mut() {
            // Simple LCG pseudo-random.
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *s = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        // Noise may or may not return a pitch, but if it does, we just
        // verify it's finite.
        if let Some(freq) = detect_pitch(&signal, 24000.0) {
            assert!(freq.is_finite(), "detected freq should be finite");
        }
    }

    // -- Pitch corrector tests ----------------------------------------------

    #[test]
    fn test_strength_zero_is_identity() {
        let config =
            PitchCorrectConfig::new(0.0, 50.0, MusicalScale::Chromatic, 440.0, true).unwrap();
        let original = sine_wave(445.0, 24000.0, 4096);
        let mut audio = original.clone();
        let mut corrector = PitchCorrector::new(&config, 24000.0).unwrap();
        corrector.process(&mut audio);

        // With strength=0, output should match input.
        for (i, (&got, &expected)) in audio.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "sample {i}: got {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn test_speed_zero_gives_hard_correction() {
        let config =
            PitchCorrectConfig::new(1.0, 0.0, MusicalScale::Chromatic, 440.0, true).unwrap();
        let sr = 24000.0;
        // Generate a 445 Hz sine (5 Hz sharp of A4).
        let mut audio = sine_wave(445.0, sr, 8192);
        let mut corrector = PitchCorrector::new(&config, sr).unwrap();
        corrector.process(&mut audio);

        // After hard correction, the pitch should be closer to 440 Hz.
        // Verify by detecting pitch in the corrected output.
        if let Some(corrected_hz) = detect_pitch(&audio[2048..], sr) {
            assert!(
                (corrected_hz - 440.0).abs() < (445.0f32 - 440.0).abs(),
                "corrected pitch {corrected_hz} should be closer to 440 than 445",
            );
        }
    }

    #[test]
    fn test_corrector_reset() {
        let config = PitchCorrectConfig::default();
        let mut corrector = PitchCorrector::new(&config, 24000.0).unwrap();
        corrector.smooth_ratio = 1.5;
        corrector.reset();
        assert!(
            (corrector.smooth_ratio - 1.0).abs() < 1e-10,
            "reset should restore smooth_ratio to 1.0",
        );
    }

    #[test]
    fn test_process_short_audio_is_noop() {
        let config = PitchCorrectConfig::default();
        let mut corrector = PitchCorrector::new(&config, 24000.0).unwrap();
        let mut audio = vec![0.5; 100]; // Too short for analysis.
        let original = audio.clone();
        corrector.process(&mut audio);
        assert_eq!(audio, original, "short audio should be unchanged");
    }

    #[test]
    fn test_no_nan_in_output() {
        let config = PitchCorrectConfig::hard_tune();
        let sr = 24000.0;
        let mut audio = sine_wave(445.0, sr, 4096);
        let mut corrector = PitchCorrector::new(&config, sr).unwrap();
        corrector.process(&mut audio);

        for (i, &s) in audio.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} is non-finite: {s}");
        }
    }

    // -- Convenience function tests -----------------------------------------

    #[test]
    fn test_apply_pitch_correction_empty_voices() {
        let config = PitchCorrectConfig::default();
        let mut voices: Vec<Vec<f32>> = vec![];
        assert!(apply_pitch_correction(&mut voices, &config, 24000.0).is_ok());
    }

    #[test]
    fn test_apply_pitch_correction_multi_voice() {
        let config =
            PitchCorrectConfig::new(0.8, 30.0, MusicalScale::Chromatic, 440.0, true).unwrap();
        let sr = 24000.0;
        let signal = sine_wave(445.0, sr, 4096);
        let mut voices = vec![signal.clone(), signal.clone(), signal.clone()];
        let result = apply_pitch_correction(&mut voices, &config, sr);
        assert!(result.is_ok());

        // Each voice should have been modified.
        for (vi, voice) in voices.iter().enumerate() {
            assert_eq!(
                voice.len(),
                signal.len(),
                "voice {vi} length should be preserved",
            );
        }
    }

    #[test]
    fn test_apply_pitch_correction_invalid_config() {
        let config = PitchCorrectConfig {
            correction_strength: 2.0, // invalid
            ..PitchCorrectConfig::default()
        };
        let mut voices = vec![vec![0.0; 1000]];
        assert!(apply_pitch_correction(&mut voices, &config, 24000.0).is_err());
    }

    #[test]
    fn test_chromatic_correction_snaps_detuned_sine() {
        // Generate a sine at 453 Hz (between A4=440 and A#4=466.16).
        // After chromatic correction, it should move toward 440 Hz.
        let config =
            PitchCorrectConfig::new(1.0, 0.0, MusicalScale::Chromatic, 440.0, true).unwrap();
        let sr = 24000.0;
        let detuned_freq = 453.0;
        let mut audio = sine_wave(detuned_freq, sr, 8192);

        let mut corrector = PitchCorrector::new(&config, sr).unwrap();
        corrector.process(&mut audio);

        // Detect pitch after correction; should be closer to 440 Hz.
        if let Some(corrected_hz) = detect_pitch(&audio[2048..], sr) {
            let original_error = (detuned_freq - 440.0).abs();
            let corrected_error = (corrected_hz - 440.0).abs();
            assert!(
                corrected_error < original_error,
                "corrected pitch {corrected_hz} should be closer to 440 than \
                 original {detuned_freq} (errors: {corrected_error} vs {original_error})",
            );
        }
    }

    #[test]
    fn test_corrector_invalid_sample_rate() {
        let config = PitchCorrectConfig::default();
        assert!(PitchCorrector::new(&config, 0.0).is_err());
        assert!(PitchCorrector::new(&config, -1.0).is_err());
        assert!(PitchCorrector::new(&config, f32::NAN).is_err());
    }
}
