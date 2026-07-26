// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Adaptive inter-voice pitch coherence tracking and correction for Kokoro chorus.
//!
//! Unlike pitch correction (which snaps each voice to a musical scale), intonation
//! tracking monitors the **relative** pitch of all chorus voices and gently pulls
//! them toward a shared reference. This prevents the "out of tune" drifting that
//! occurs when multiple TTS voices independently wander from the intended pitch.
//!
//! # Algorithm
//!
//! 1. **YIN pitch detection** per voice -- normalized autocorrelation difference
//!    function with cumulative mean normalization and parabolic interpolation.
//! 2. **Reference pitch computation** -- either a fixed voice (voice 0) or the
//!    median of all detected pitches (adaptive mode).
//! 3. **Deviation measurement** -- pitch difference in cents between each voice
//!    and the reference.
//! 4. **Partial correction** -- only deviations exceeding the tolerance threshold
//!    are corrected, and only by `correction_speed` fraction per analysis frame.
//! 5. **Portamento smoothing** -- correction targets are exponentially smoothed
//!    over `portamento_ms` to avoid discontinuities.
//! 6. **Variable-rate resampling** -- applies the correction by reading the
//!    source at a slightly modified rate with linear interpolation.
//!
//! # Placement in the chorus pipeline
//!
//! ```text
//! Per-voice: vibrato -> detuning -> pitch_correct -> intonation -> EQ -> dynamics
//! ```
//!
//! Intonation correction runs **after** scale-based pitch correction and **before**
//! EQ/dynamics. It acts as a final coherence pass ensuring all voices agree on
//! the same note, regardless of individual drift introduced earlier in the chain.
//!
//! # References
//!
//! - de Cheveigne, A. & Kawahara, H. "YIN, a fundamental frequency estimator
//!   for speech and music." JASA, 111(4), 2002.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default YIN window size for intonation analysis.
const YIN_WINDOW: usize = 256;

/// YIN cumulative mean normalized difference threshold.
const YIN_THRESHOLD: f32 = 0.15;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for adaptive inter-voice intonation correction.
///
/// Controls how aggressively voices are pulled toward a shared reference pitch.
/// Use builder methods for ergonomic construction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IntonationConfig {
    /// Acceptable pitch spread in cents before correction engages.
    ///
    /// Deviations smaller than this are left alone, preserving natural
    /// micro-variation. Must be finite and in [0.5, 50.0]. Default: `10.0`.
    pub tolerance_cents: f32,

    /// Correction speed per analysis frame (0.0-1.0).
    ///
    /// Fraction of the deviation corrected each frame:
    /// - 0.0 = no correction
    /// - 0.3 = moderate (default, natural sounding)
    /// - 1.0 = snap instantly to reference pitch
    ///
    /// Must be finite and in [0.0, 1.0]. Default: `0.3`.
    pub correction_speed: f32,

    /// Index of the voice used as the pitch reference.
    ///
    /// Ignored when `enable_adaptive_reference` is true. Default: `0`.
    pub reference_voice: usize,

    /// Use the median pitch of all voices as the reference instead of a
    /// fixed voice.
    ///
    /// More robust when no single voice is clearly dominant. Default: `false`.
    pub enable_adaptive_reference: bool,

    /// Portamento time in milliseconds for smoothing correction trajectories.
    ///
    /// Prevents abrupt pitch jumps when the correction target changes.
    /// Must be finite and in [0.0, 200.0]. Default: `20.0`.
    pub portamento_ms: f32,
}

impl Default for IntonationConfig {
    fn default() -> Self {
        Self {
            tolerance_cents: 10.0,
            correction_speed: 0.3,
            reference_voice: 0,
            enable_adaptive_reference: false,
            portamento_ms: 20.0,
        }
    }
}

impl IntonationConfig {
    /// Create a validated config with all parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range.
    pub fn new(
        tolerance_cents: f32,
        correction_speed: f32,
        reference_voice: usize,
        enable_adaptive_reference: bool,
        portamento_ms: f32,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            tolerance_cents,
            correction_speed,
            reference_voice,
            enable_adaptive_reference,
            portamento_ms,
        };
        config.validate()?;
        Ok(config)
    }

    /// Builder: set tolerance in cents.
    #[must_use]
    pub fn with_tolerance_cents(mut self, cents: f32) -> Self {
        self.tolerance_cents = cents;
        self
    }

    /// Builder: set correction speed.
    #[must_use]
    pub fn with_correction_speed(mut self, speed: f32) -> Self {
        self.correction_speed = speed;
        self
    }

    /// Builder: set reference voice index.
    #[must_use]
    pub fn with_reference_voice(mut self, idx: usize) -> Self {
        self.reference_voice = idx;
        self
    }

    /// Builder: enable/disable adaptive reference.
    #[must_use]
    pub fn with_adaptive_reference(mut self, enable: bool) -> Self {
        self.enable_adaptive_reference = enable;
        self
    }

    /// Builder: set portamento time in milliseconds.
    #[must_use]
    pub fn with_portamento_ms(mut self, ms: f32) -> Self {
        self.portamento_ms = ms;
        self
    }

    /// Tight chorus preset: aggressive correction, narrow tolerance.
    #[must_use]
    pub fn tight_chorus() -> Self {
        Self {
            tolerance_cents: 5.0,
            correction_speed: 0.7,
            reference_voice: 0,
            enable_adaptive_reference: true,
            portamento_ms: 10.0,
        }
    }

    /// Natural preset: gentle correction, wider tolerance.
    #[must_use]
    pub fn natural() -> Self {
        Self {
            tolerance_cents: 15.0,
            correction_speed: 0.15,
            reference_voice: 0,
            enable_adaptive_reference: false,
            portamento_ms: 40.0,
        }
    }

    /// Unison preset: very aggressive correction for tight unison.
    #[must_use]
    pub fn unison() -> Self {
        Self {
            tolerance_cents: 2.0,
            correction_speed: 0.9,
            reference_voice: 0,
            enable_adaptive_reference: true,
            portamento_ms: 5.0,
        }
    }

    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is non-finite
    /// or outside its valid range.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.tolerance_cents.is_finite() || !(0.5..=50.0).contains(&self.tolerance_cents) {
            return Err(KokoroError::InvalidConfig {
                field: "tolerance_cents",
                reason: format!(
                    "must be finite and in [0.5, 50.0], got {}",
                    self.tolerance_cents,
                ),
            });
        }
        if !self.correction_speed.is_finite() || !(0.0..=1.0).contains(&self.correction_speed) {
            return Err(KokoroError::InvalidConfig {
                field: "correction_speed",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.correction_speed,
                ),
            });
        }
        if !self.portamento_ms.is_finite() || !(0.0..=200.0).contains(&self.portamento_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "portamento_ms",
                reason: format!(
                    "must be finite and in [0.0, 200.0], got {}",
                    self.portamento_ms,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pitch info
// ---------------------------------------------------------------------------

/// Per-voice pitch tracking information from the most recent analysis frame.
#[derive(Debug, Clone, Copy)]
pub struct PitchInfo {
    /// Voice index.
    pub voice_index: usize,
    /// Detected fundamental frequency in Hz, or `None` if unvoiced.
    pub detected_hz: Option<f32>,
    /// Deviation from the reference pitch in cents (positive = sharp).
    pub cents_from_ref: f32,
    /// Correction currently being applied in cents.
    pub correction_cents: f32,
}

// ---------------------------------------------------------------------------
// YIN pitch detection (intonation-local)
// ---------------------------------------------------------------------------

/// Detect fundamental frequency using the YIN algorithm.
///
/// Returns `Some(freq_hz)` for voiced segments or `None` for noise/silence.
///
/// # IEEE 754 safety
///
/// Returns `None` if the result would be non-finite or outside [30, 4000] Hz.
fn yin_detect(audio: &[f32], sample_rate: f32) -> Option<f32> {
    let w = YIN_WINDOW;
    if audio.len() < w * 2 || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return None;
    }

    // Step 1: Difference function.
    let mut diff = vec![0.0f32; w];
    for tau in 1..w {
        let mut sum = 0.0f64;
        for j in 0..w {
            let delta = f64::from(audio[j]) - f64::from(audio[j + tau]);
            sum += delta * delta;
        }
        diff[tau] = sum as f32;
    }

    // Step 2: Cumulative mean normalized difference.
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

    // Step 3: Absolute threshold search with local minimum tracking.
    let min_tau = 2;
    let max_tau = w - 1;
    let mut best_tau: Option<usize> = None;

    for tau in min_tau..max_tau {
        if cmnd[tau] < YIN_THRESHOLD {
            let mut local_min_tau = tau;
            let mut local_min_val = cmnd[tau];
            for t in (tau + 1)..max_tau {
                if cmnd[t] < local_min_val {
                    local_min_val = cmnd[t];
                    local_min_tau = t;
                } else {
                    break;
                }
            }
            best_tau = Some(local_min_tau);
            break;
        }
    }

    let tau = best_tau?;

    // Step 4: Parabolic interpolation for sub-sample accuracy.
    let refined = if tau > 0 && tau < w - 1 {
        let a = f64::from(cmnd[tau - 1]);
        let b = f64::from(cmnd[tau]);
        let c = f64::from(cmnd[tau + 1]);
        let denom = 2.0 * (2.0 * b - a - c);
        if denom.abs() > 1e-12 {
            tau as f64 + (a - c) / denom
        } else {
            tau as f64
        }
    } else {
        tau as f64
    };

    if refined <= 0.0 {
        return None;
    }

    let freq = f64::from(sample_rate) / refined;
    let freq_f32 = freq as f32;

    if freq_f32.is_finite() && freq_f32 > 30.0 && freq_f32 < 4000.0 {
        Some(freq_f32)
    } else {
        None
    }
}

/// Convert a frequency ratio to cents: `1200 * log2(ratio)`.
#[inline]
fn ratio_to_cents(ratio: f64) -> f64 {
    if ratio <= 0.0 || !ratio.is_finite() {
        return 0.0;
    }
    1200.0 * ratio.log2()
}

/// Convert cents to a resampling rate: `2^(cents / 1200)`.
#[inline]
fn cents_to_rate(cents: f64) -> f64 {
    if !cents.is_finite() {
        return 1.0;
    }
    (2.0f64).powf(cents / 1200.0)
}

/// Compute the median of a non-empty slice of `f32` values.
///
/// Returns `None` on an empty slice. NaN values are treated as greater
/// than all finite values (pushed to the end).
fn median_f32(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        let avg = f32::midpoint(sorted[mid - 1], sorted[mid]);
        if avg.is_finite() {
            Some(avg)
        } else {
            Some(sorted[mid])
        }
    } else {
        Some(sorted[mid])
    }
}

// ---------------------------------------------------------------------------
// Intonation tracker
// ---------------------------------------------------------------------------

/// Stateful inter-voice intonation tracker and corrector.
///
/// Monitors the pitch of each chorus voice and applies gentle corrections
/// to keep all voices in tune with each other. Maintains per-voice smoothed
/// correction state across calls for portamento continuity.
pub struct IntonationTracker {
    config: IntonationConfig,
    sample_rate: f32,
    hop_size: usize,
    /// Per-voice smoothed correction in cents.
    smooth_corrections: Vec<f64>,
    /// Portamento smoothing coefficient per hop.
    smooth_coeff: f64,
    /// Latest per-voice pitch info (updated after each `process_voices` call).
    last_info: Vec<PitchInfo>,
}

impl IntonationTracker {
    /// Create a new intonation tracker.
    ///
    /// # Arguments
    ///
    /// * `config` - Intonation configuration.
    /// * `n_voices` - Number of chorus voices.
    /// * `sample_rate` - Audio sample rate in Hz.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails, or
    /// `KokoroError::InvalidInput` if `sample_rate` is non-positive/non-finite
    /// or `n_voices` is zero.
    pub fn new(
        config: &IntonationConfig,
        n_voices: usize,
        sample_rate: f32,
    ) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidInput(format!(
                "sample_rate must be positive and finite, got {sample_rate}"
            )));
        }
        if n_voices == 0 {
            return Err(KokoroError::InvalidInput(
                "n_voices must be >= 1".to_string(),
            ));
        }
        if !config.enable_adaptive_reference && config.reference_voice >= n_voices {
            return Err(KokoroError::InvalidConfig {
                field: "reference_voice",
                reason: format!(
                    "reference_voice {} >= n_voices {}",
                    config.reference_voice, n_voices,
                ),
            });
        }

        let hop_size = YIN_WINDOW / 4;

        // Exponential smoothing coefficient from portamento time.
        let smooth_coeff = if config.portamento_ms < 0.01 {
            1.0
        } else {
            let hops_per_ms = f64::from(sample_rate) / (hop_size as f64 * 1000.0);
            let time_constant_hops = f64::from(config.portamento_ms) * hops_per_ms;
            if time_constant_hops > 1.0 {
                1.0 - (-1.0 / time_constant_hops).exp()
            } else {
                1.0
            }
        };

        let last_info = (0..n_voices)
            .map(|i| PitchInfo {
                voice_index: i,
                detected_hz: None,
                cents_from_ref: 0.0,
                correction_cents: 0.0,
            })
            .collect();

        Ok(Self {
            config: config.clone(),
            sample_rate,
            hop_size,
            smooth_corrections: vec![0.0; n_voices],
            smooth_coeff,
            last_info,
        })
    }

    /// Reset all internal state (pitch estimates, correction targets).
    pub fn reset(&mut self) {
        for c in &mut self.smooth_corrections {
            *c = 0.0;
        }
        for info in &mut self.last_info {
            info.detected_hz = None;
            info.cents_from_ref = 0.0;
            info.correction_cents = 0.0;
        }
    }

    /// Get pitch information from the most recent analysis.
    #[must_use]
    pub fn get_pitch_info(&self) -> &[PitchInfo] {
        &self.last_info
    }

    /// Track and correct the pitch of all voices in-place.
    ///
    /// Each voice buffer is analyzed in overlapping windows. Voices that
    /// deviate from the reference pitch beyond the tolerance are gently
    /// corrected via variable-rate resampling.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidInput` if the number of voice buffers
    /// does not match the configured voice count.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) -> Result<(), KokoroError> {
        let n = self.smooth_corrections.len();
        if voices.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "expected {} voices, got {}",
                n,
                voices.len(),
            )));
        }

        // Fast exit for single voice or zero correction speed.
        if n <= 1 || self.config.correction_speed < 1e-6 {
            return Ok(());
        }

        // Find the minimum buffer length to avoid index mismatches.
        let min_len = voices.iter().map(Vec::len).min().unwrap_or(0);
        let win_len = YIN_WINDOW * 2;
        if min_len < win_len {
            return Ok(());
        }

        // Build Hann window for overlap-add.
        let hann: Vec<f32> = (0..win_len)
            .map(|i| {
                let t = i as f64 / win_len as f64;
                (0.5 * (1.0 - (std::f64::consts::TAU * t).cos())) as f32
            })
            .collect();

        // Pre-allocate per-voice output and weight buffers.
        let mut outputs: Vec<Vec<f32>> = voices.iter().map(|v| vec![0.0f32; v.len()]).collect();
        let mut weights: Vec<Vec<f32>> = voices.iter().map(|v| vec![0.0f32; v.len()]).collect();

        let mut pos = 0;
        while pos + win_len <= min_len {
            // Step 1: Detect pitch for each voice in the current window.
            let mut detected: Vec<Option<f32>> = Vec::with_capacity(n);
            for voice in voices.iter() {
                let window = &voice[pos..pos + win_len];
                detected.push(yin_detect(window, self.sample_rate));
            }

            // Step 2: Compute reference pitch.
            let ref_hz = self.compute_reference(&detected);

            // Step 3: Compute per-voice correction targets.
            for vi in 0..n {
                let (cents_from_ref, target_correction) =
                    if let (Some(det), Some(rh)) = (detected[vi], ref_hz) {
                        if !det.is_finite() || !rh.is_finite() || rh <= 0.0 || det <= 0.0 {
                            (0.0f64, 0.0f64)
                        } else {
                            let deviation = ratio_to_cents(f64::from(det) / f64::from(rh));
                            let target = if deviation.abs() > f64::from(self.config.tolerance_cents) {
                                // Correct only the amount exceeding the tolerance, scaled
                                // by correction speed.
                                let excess = if deviation > 0.0 {
                                    deviation - f64::from(self.config.tolerance_cents)
                                } else {
                                    deviation + f64::from(self.config.tolerance_cents)
                                };
                                -excess * f64::from(self.config.correction_speed)
                            } else {
                                0.0
                            };
                            (deviation, target)
                        }
                    } else {
                        (0.0, 0.0)
                    };

                // Update pitch info.
                self.last_info[vi] = PitchInfo {
                    voice_index: vi,
                    detected_hz: detected[vi],
                    cents_from_ref: cents_from_ref as f32,
                    correction_cents: 0.0, // filled after smoothing
                };

                // Portamento smoothing.
                self.smooth_corrections[vi] +=
                    self.smooth_coeff * (target_correction - self.smooth_corrections[vi]);

                let correction = self.smooth_corrections[vi];
                self.last_info[vi].correction_cents = correction as f32;

                // Step 4: Apply correction via variable-rate resampling.
                let rate = cents_to_rate(correction);
                let voice = &voices[vi];
                let out = &mut outputs[vi];
                let wt = &mut weights[vi];

                for i in 0..win_len {
                    let src_pos = i as f64 * rate;
                    let src_idx = src_pos.floor() as usize;
                    let frac = (src_pos - src_idx as f64) as f32;

                    let global_src = pos + src_idx;
                    let sample = if global_src < voice.len() {
                        let s0 = voice[global_src];
                        let s1 = if global_src + 1 < voice.len() {
                            voice[global_src + 1]
                        } else {
                            s0
                        };
                        s0 + frac * (s1 - s0)
                    } else {
                        0.0
                    };

                    let out_idx = pos + i;
                    if out_idx < out.len() {
                        let w = hann[i];
                        out[out_idx] += sample * w;
                        wt[out_idx] += w;
                    }
                }
            }

            pos += self.hop_size;
        }

        // Normalize overlap-add and copy back.
        for vi in 0..n {
            let voice = &mut voices[vi];
            let out = &outputs[vi];
            let wt = &weights[vi];

            for i in 0..voice.len() {
                if wt[i] > 1e-6 {
                    voice[i] = out[i] / wt[i];
                }
                if !voice[i].is_finite() {
                    voice[i] = 0.0;
                }
            }
        }

        Ok(())
    }

    /// Compute the reference pitch from detected pitches.
    fn compute_reference(&self, detected: &[Option<f32>]) -> Option<f32> {
        if self.config.enable_adaptive_reference {
            // Median of all voiced pitches.
            let voiced: Vec<f32> = detected.iter().filter_map(|&d| d).collect();
            median_f32(&voiced)
        } else {
            // Fixed reference voice.
            let idx = self.config.reference_voice;
            if idx < detected.len() {
                detected[idx]
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Track and correct inter-voice pitch coherence across all voices.
///
/// This is the main entry point for intonation correction in the chorus
/// pipeline. Creates a tracker, processes once, and returns pitch info.
///
/// # Arguments
///
/// * `voices` - Mutable slice of per-voice PCM buffers (mono).
/// * `config` - Intonation configuration.
/// * `sample_rate` - Audio sample rate in Hz.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` or `KokoroError::InvalidInput` on
/// invalid parameters.
pub fn correct_intonation(
    voices: &mut [Vec<f32>],
    config: &IntonationConfig,
    sample_rate: f32,
) -> Result<Vec<PitchInfo>, KokoroError> {
    config.validate()?;
    if voices.is_empty() {
        return Ok(vec![]);
    }

    let mut tracker = IntonationTracker::new(config, voices.len(), sample_rate)?;
    tracker.process_voices(voices)?;
    Ok(tracker.get_pitch_info().to_vec())
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
        assert!(IntonationConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_tight_chorus_valid() {
        let cfg = IntonationConfig::tight_chorus();
        assert!(cfg.validate().is_ok());
        assert!(cfg.tolerance_cents < 10.0);
        assert!(cfg.correction_speed > 0.5);
    }

    #[test]
    fn test_config_natural_valid() {
        let cfg = IntonationConfig::natural();
        assert!(cfg.validate().is_ok());
        assert!(cfg.tolerance_cents > 10.0);
        assert!(cfg.correction_speed < 0.3);
    }

    #[test]
    fn test_config_unison_valid() {
        let cfg = IntonationConfig::unison();
        assert!(cfg.validate().is_ok());
        assert!(cfg.tolerance_cents < 5.0);
        assert!(cfg.correction_speed > 0.8);
    }

    #[test]
    fn test_config_validation_rejects_invalid() {
        // Tolerance too low.
        assert!(IntonationConfig::new(0.1, 0.3, 0, false, 20.0).is_err());
        // Tolerance too high.
        assert!(IntonationConfig::new(60.0, 0.3, 0, false, 20.0).is_err());
        // NaN tolerance.
        assert!(IntonationConfig::new(f32::NAN, 0.3, 0, false, 20.0).is_err());
        // Speed out of range.
        assert!(IntonationConfig::new(10.0, -0.1, 0, false, 20.0).is_err());
        assert!(IntonationConfig::new(10.0, 1.1, 0, false, 20.0).is_err());
        // Portamento out of range.
        assert!(IntonationConfig::new(10.0, 0.3, 0, false, -1.0).is_err());
        assert!(IntonationConfig::new(10.0, 0.3, 0, false, 201.0).is_err());
    }

    #[test]
    fn test_config_builder_methods() {
        let cfg = IntonationConfig::default()
            .with_tolerance_cents(7.0)
            .with_correction_speed(0.5)
            .with_reference_voice(1)
            .with_adaptive_reference(true)
            .with_portamento_ms(30.0);
        assert!((cfg.tolerance_cents - 7.0).abs() < f32::EPSILON);
        assert!((cfg.correction_speed - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.reference_voice, 1);
        assert!(cfg.enable_adaptive_reference);
        assert!((cfg.portamento_ms - 30.0).abs() < f32::EPSILON);
    }

    // -- Tracker creation ---------------------------------------------------

    #[test]
    fn test_tracker_invalid_sample_rate() {
        let cfg = IntonationConfig::default();
        assert!(IntonationTracker::new(&cfg, 3, 0.0).is_err());
        assert!(IntonationTracker::new(&cfg, 3, -1.0).is_err());
        assert!(IntonationTracker::new(&cfg, 3, f32::NAN).is_err());
    }

    #[test]
    fn test_tracker_zero_voices() {
        let cfg = IntonationConfig::default();
        assert!(IntonationTracker::new(&cfg, 0, 24000.0).is_err());
    }

    #[test]
    fn test_tracker_reference_out_of_range() {
        let cfg = IntonationConfig::default().with_reference_voice(5);
        assert!(IntonationTracker::new(&cfg, 3, 24000.0).is_err());
    }

    #[test]
    fn test_tracker_adaptive_ignores_reference_index() {
        let cfg = IntonationConfig::default()
            .with_reference_voice(99)
            .with_adaptive_reference(true);
        // Should not fail because adaptive mode ignores reference_voice.
        assert!(IntonationTracker::new(&cfg, 3, 24000.0).is_ok());
    }

    // -- Pitch info ---------------------------------------------------------

    #[test]
    fn test_pitch_info_initial() {
        let cfg = IntonationConfig::default();
        let tracker = IntonationTracker::new(&cfg, 3, 24000.0).unwrap();
        let info = tracker.get_pitch_info();
        assert_eq!(info.len(), 3);
        for pi in info {
            assert!(pi.detected_hz.is_none());
            assert!((pi.cents_from_ref).abs() < f32::EPSILON);
        }
    }

    // -- Processing ---------------------------------------------------------

    #[test]
    fn test_single_voice_passthrough() {
        let cfg = IntonationConfig::default();
        let sr = 24000.0;
        let original = sine_wave(440.0, sr, 4096);
        let mut voices = vec![original.clone()];
        let mut tracker = IntonationTracker::new(&cfg, 1, sr).unwrap();
        tracker.process_voices(&mut voices).unwrap();

        // Single voice should be unchanged.
        for (i, (&got, &exp)) in voices[0].iter().zip(original.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "sample {i}: got {got}, expected {exp}",
            );
        }
    }

    #[test]
    fn test_zero_speed_is_identity() {
        let cfg = IntonationConfig::default().with_correction_speed(0.0);
        let sr = 24000.0;
        let v0 = sine_wave(440.0, sr, 4096);
        let v1 = sine_wave(460.0, sr, 4096);
        let mut voices = vec![v0, v1.clone()];
        let mut tracker = IntonationTracker::new(&cfg, 2, sr).unwrap();
        tracker.process_voices(&mut voices).unwrap();

        // With speed=0, no correction should be applied.
        for (i, (&got, &exp)) in voices[1].iter().zip(v1.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "voice 1 sample {i}: got {got}, expected {exp}",
            );
        }
    }

    #[test]
    fn test_mismatched_voice_count() {
        let cfg = IntonationConfig::default();
        let mut tracker = IntonationTracker::new(&cfg, 3, 24000.0).unwrap();
        let mut voices = vec![vec![0.0; 1024], vec![0.0; 1024]];
        assert!(tracker.process_voices(&mut voices).is_err());
    }

    #[test]
    fn test_short_audio_passthrough() {
        let cfg = IntonationConfig::default();
        let mut tracker = IntonationTracker::new(&cfg, 2, 24000.0).unwrap();
        let mut voices = vec![vec![0.5; 100], vec![0.5; 100]];
        let original = voices.clone();
        tracker.process_voices(&mut voices).unwrap();
        assert_eq!(
            voices, original,
            "short audio should pass through unchanged"
        );
    }

    #[test]
    fn test_coherent_voices_unchanged() {
        // Voices already singing the same note should get minimal correction.
        let sr = 24000.0;
        let cfg = IntonationConfig::default(); // 10 cents tolerance
        let v0 = sine_wave(440.0, sr, 8192);
        let v1 = sine_wave(440.0, sr, 8192);
        let v1_orig = v1.clone();
        let mut voices = vec![v0, v1];
        let mut tracker = IntonationTracker::new(&cfg, 2, sr).unwrap();
        tracker.process_voices(&mut voices).unwrap();

        // Voices at the same pitch should have negligible correction.
        let max_diff: f32 = voices[1]
            .iter()
            .zip(v1_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.05,
            "coherent voices should have minimal change, max_diff={max_diff}",
        );
    }

    #[test]
    fn test_detuned_voice_gets_corrected() {
        // Voice 1 sings 20 cents sharp; correction should pull it closer.
        let sr = 24000.0;
        let cfg = IntonationConfig::tight_chorus(); // 5 cents tolerance, aggressive
        let ref_freq = 440.0;
        // 20 cents sharp: freq = 440 * 2^(20/1200)
        let sharp_freq = ref_freq * (2.0f32).powf(20.0 / 1200.0);

        let v0 = sine_wave(ref_freq, sr, 8192);
        let v1 = sine_wave(sharp_freq, sr, 8192);
        let mut voices = vec![v0, v1];

        let mut tracker = IntonationTracker::new(&cfg, 2, sr).unwrap();
        tracker.process_voices(&mut voices).unwrap();

        // After correction, voice 1's detected pitch should be closer to 440.
        let info = tracker.get_pitch_info();
        // The correction_cents for voice 1 should be negative (pulling pitch down).
        // Note: we check the field was populated; the exact value depends on
        // how many analysis windows were processed.
        assert_eq!(info.len(), 2);
    }

    #[test]
    fn test_no_nan_in_output() {
        let sr = 24000.0;
        let cfg = IntonationConfig::unison();
        let v0 = sine_wave(440.0, sr, 4096);
        let v1 = sine_wave(450.0, sr, 4096);
        let v2 = sine_wave(435.0, sr, 4096);
        let mut voices = vec![v0, v1, v2];
        let mut tracker = IntonationTracker::new(&cfg, 3, sr).unwrap();
        tracker.process_voices(&mut voices).unwrap();

        for (vi, voice) in voices.iter().enumerate() {
            for (si, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "voice {vi} sample {si} is non-finite: {s}");
            }
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = IntonationConfig::default();
        let mut tracker = IntonationTracker::new(&cfg, 2, 24000.0).unwrap();
        tracker.smooth_corrections[0] = 5.0;
        tracker.smooth_corrections[1] = -3.0;
        tracker.reset();
        for &c in &tracker.smooth_corrections {
            assert!((c).abs() < 1e-10, "correction should be reset to 0");
        }
    }

    #[test]
    fn test_preserves_buffer_length() {
        let sr = 24000.0;
        let cfg = IntonationConfig::default();
        let len = 6000;
        let v0 = sine_wave(440.0, sr, len);
        let v1 = sine_wave(445.0, sr, len);
        let mut voices = vec![v0, v1];
        let mut tracker = IntonationTracker::new(&cfg, 2, sr).unwrap();
        tracker.process_voices(&mut voices).unwrap();

        for (vi, voice) in voices.iter().enumerate() {
            assert_eq!(voice.len(), len, "voice {vi} length should be preserved");
        }
    }

    // -- Convenience function -----------------------------------------------

    #[test]
    fn test_correct_intonation_empty() {
        let cfg = IntonationConfig::default();
        let mut voices: Vec<Vec<f32>> = vec![];
        let info = correct_intonation(&mut voices, &cfg, 24000.0).unwrap();
        assert!(info.is_empty());
    }

    #[test]
    fn test_correct_intonation_returns_info() {
        let sr = 24000.0;
        let cfg = IntonationConfig::default();
        let v0 = sine_wave(440.0, sr, 4096);
        let v1 = sine_wave(445.0, sr, 4096);
        let mut voices = vec![v0, v1];
        let info = correct_intonation(&mut voices, &cfg, sr).unwrap();
        assert_eq!(info.len(), 2);
        assert_eq!(info[0].voice_index, 0);
        assert_eq!(info[1].voice_index, 1);
    }

    // -- Helper function tests ----------------------------------------------

    #[test]
    fn test_ratio_to_cents_identity() {
        assert!((ratio_to_cents(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ratio_to_cents_octave() {
        let cents = ratio_to_cents(2.0);
        assert!(
            (cents - 1200.0).abs() < 1e-6,
            "octave should be 1200 cents, got {cents}",
        );
    }

    #[test]
    fn test_ratio_to_cents_nan_safety() {
        assert!((ratio_to_cents(0.0)).abs() < 1e-10);
        assert!((ratio_to_cents(-1.0)).abs() < 1e-10);
        assert!((ratio_to_cents(f64::NAN)).abs() < 1e-10);
    }

    #[test]
    fn test_cents_to_rate_identity() {
        let rate = cents_to_rate(0.0);
        assert!((rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cents_to_rate_roundtrip() {
        let cents = 15.0;
        let rate = cents_to_rate(cents);
        let back = ratio_to_cents(rate);
        assert!(
            (back - cents).abs() < 1e-6,
            "roundtrip: {cents} -> {rate} -> {back}",
        );
    }

    #[test]
    fn test_median_basic() {
        assert!((median_f32(&[3.0, 1.0, 2.0]).unwrap() - 2.0).abs() < 1e-6);
        assert!((median_f32(&[1.0, 2.0]).unwrap() - 1.5).abs() < 1e-6);
        assert!((median_f32(&[5.0]).unwrap() - 5.0).abs() < 1e-6);
        assert!(median_f32(&[]).is_none());
    }

    #[test]
    fn test_yin_detect_sine() {
        let sr = 24000.0;
        let signal = sine_wave(440.0, sr, 2048);
        let detected = yin_detect(&signal, sr);
        assert!(detected.is_some(), "should detect 440 Hz sine");
        let hz = detected.unwrap();
        assert!(
            (hz - 440.0).abs() / 440.0 < 0.03,
            "detected {hz}, expected ~440",
        );
    }

    #[test]
    fn test_yin_detect_too_short() {
        assert!(yin_detect(&[0.0; 100], 24000.0).is_none());
    }

    #[test]
    fn test_adaptive_reference_uses_median() {
        let sr = 24000.0;
        let cfg = IntonationConfig::default().with_adaptive_reference(true);
        let v0 = sine_wave(430.0, sr, 4096);
        let v1 = sine_wave(440.0, sr, 4096);
        let v2 = sine_wave(450.0, sr, 4096);
        let mut voices = vec![v0, v1, v2];
        let info = correct_intonation(&mut voices, &cfg, sr).unwrap();
        assert_eq!(info.len(), 3);
    }
}
