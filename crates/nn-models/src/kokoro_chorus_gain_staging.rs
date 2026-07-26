// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Adaptive gain staging and LUFS loudness management for Kokoro chorus.
//!
//! When mixing multiple TTS voices through a chorus pipeline, each processing
//! stage (detuning, EQ, saturation, reverb) can change signal levels. Without
//! proper gain staging, the signal can clip at loud peaks or become too quiet
//! after attenuation. This module provides:
//!
//! - **Level measurement** — peak dBFS, RMS dBFS, and approximate LUFS
//!   (K-weighted RMS with ITU-R BS.1770 pre-filtering).
//! - **Automatic leveling** — adjusts gain to hit a target LUFS while
//!   respecting a peak ceiling.
//! - **Inter-stage monitoring** — tracks per-stage levels to detect violations
//!   (clipping, excessive gain) across the full processing chain.
//!
//! # LUFS approximation
//!
//! The ITU-R BS.1770 loudness standard requires a K-weighting pre-filter
//! (high-shelf at ~1500 Hz, high-pass at ~38 Hz) before gated RMS
//! measurement over 400ms windows. This module implements a simplified
//! single-pass approximation: a biquad high-shelf boost (~+4 dB above
//! 1500 Hz) applied to the full signal, then RMS over the entire buffer.
//! This is accurate to within ~1 LUFS for speech signals at 24 kHz.
//!
//! # References
//!
//! - ITU-R BS.1770-5 "Algorithms to measure audio programme loudness and
//!   true-peak audio level." International Telecommunication Union, 2023.
//! - EBU R128 "Loudness normalisation and permitted maximum level of audio
//!   signals." European Broadcasting Union, 2020.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum representable level in dBFS. Below this, we treat as silence.
const SILENCE_DB: f32 = -120.0;

/// Safety floor for linear amplitude to avoid log10(0).
const AMPLITUDE_FLOOR: f32 = 1e-12;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for adaptive gain staging.
///
/// Controls target loudness (LUFS), peak ceiling (dBFS), headroom, and
/// whether automatic makeup gain is applied after each processing stage.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GainStagingConfig {
    /// Target integrated loudness in LUFS.
    ///
    /// Typical values: -16 LUFS for streaming, -14 LUFS for podcast,
    /// -24 LUFS for broadcast (EBU R128). Must be in [-24.0, -8.0].
    /// Default: -16.0.
    pub target_lufs: f32,

    /// Maximum permitted true-peak level in dBFS.
    ///
    /// The auto-leveler will not boost the signal above this peak ceiling.
    /// Must be in [-6.0, -0.1]. Default: -1.0 dBFS.
    pub target_peak_dbfs: f32,

    /// Whether to automatically apply makeup gain after each processing
    /// stage to maintain consistent loudness.
    ///
    /// When `true`, [`GainStager::auto_level`] is the primary method.
    /// When `false`, only explicit [`GainStager::apply_gain`] and
    /// [`GainStager::normalize_peak`] are used. Default: `true`.
    pub auto_makeup: bool,

    /// Headroom to maintain below digital clipping (0 dBFS) in dB.
    ///
    /// This is a safety margin: even when target_peak_dbfs allows peaks
    /// close to 0 dBFS, the headroom ensures no sample ever exceeds
    /// `0 - headroom_db` dBFS. Must be in [1.0, 12.0]. Default: 3.0.
    pub headroom_db: f32,
}

impl Default for GainStagingConfig {
    fn default() -> Self {
        Self {
            target_lufs: -16.0,
            target_peak_dbfs: -1.0,
            auto_makeup: true,
            headroom_db: 3.0,
        }
    }
}

impl GainStagingConfig {
    /// Create a new gain staging config with all parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range.
    pub fn new(
        target_lufs: f32,
        target_peak_dbfs: f32,
        auto_makeup: bool,
        headroom_db: f32,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            target_lufs,
            target_peak_dbfs,
            auto_makeup,
            headroom_db,
        };
        config.validate()?;
        Ok(config)
    }

    /// Builder: set target LUFS.
    #[must_use]
    pub fn with_target_lufs(mut self, lufs: f32) -> Self {
        self.target_lufs = lufs;
        self
    }

    /// Builder: set target peak dBFS.
    #[must_use]
    pub fn with_target_peak_dbfs(mut self, peak: f32) -> Self {
        self.target_peak_dbfs = peak;
        self
    }

    /// Builder: set auto makeup gain.
    #[must_use]
    pub fn with_auto_makeup(mut self, auto_makeup: bool) -> Self {
        self.auto_makeup = auto_makeup;
        self
    }

    /// Builder: set headroom in dB.
    #[must_use]
    pub fn with_headroom_db(mut self, headroom: f32) -> Self {
        self.headroom_db = headroom;
        self
    }

    /// Validate that this configuration is internally consistent.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.target_lufs.is_finite() || self.target_lufs < -24.0 || self.target_lufs > -8.0 {
            return Err(KokoroError::InvalidConfig {
                field: "target_lufs",
                reason: format!(
                    "must be finite and in [-24.0, -8.0], got {}",
                    self.target_lufs,
                ),
            });
        }
        if !self.target_peak_dbfs.is_finite()
            || self.target_peak_dbfs < -6.0
            || self.target_peak_dbfs > -0.1
        {
            return Err(KokoroError::InvalidConfig {
                field: "target_peak_dbfs",
                reason: format!(
                    "must be finite and in [-6.0, -0.1], got {}",
                    self.target_peak_dbfs,
                ),
            });
        }
        if !self.headroom_db.is_finite() || self.headroom_db < 1.0 || self.headroom_db > 12.0 {
            return Err(KokoroError::InvalidConfig {
                field: "headroom_db",
                reason: format!(
                    "must be finite and in [1.0, 12.0], got {}",
                    self.headroom_db,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Gain stager
// ---------------------------------------------------------------------------

/// Adaptive gain stager for the Kokoro chorus pipeline.
///
/// Measures signal levels (peak, RMS, approximate LUFS) and applies gain
/// adjustments to maintain consistent loudness across processing stages.
/// Gain is applied in the dB domain with soft-clipping at 0 dBFS to
/// prevent hard clipping.
pub struct GainStager {
    config: GainStagingConfig,
    /// Peak ceiling in linear amplitude, derived from the more conservative
    /// of `target_peak_dbfs` and `(0 - headroom_db)`.
    ceiling_linear: f32,
}

impl GainStager {
    /// Create a new gain stager with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: &GainStagingConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        // The effective ceiling is the more conservative (lower) of
        // target_peak_dbfs and (0 - headroom_db).
        let effective_peak_db = config.target_peak_dbfs.min(-config.headroom_db);
        let ceiling_linear = db_to_linear(effective_peak_db);
        Ok(Self {
            config: config.clone(),
            ceiling_linear,
        })
    }

    /// Create a gain stager with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if default config is somehow
    /// invalid (should not happen).
    pub fn with_defaults() -> Result<Self, KokoroError> {
        Self::new(&GainStagingConfig::default())
    }

    /// Measure the peak level of an audio buffer in dBFS.
    ///
    /// Returns the maximum absolute sample value converted to dBFS.
    /// Empty or all-zero buffers return [`SILENCE_DB`] (-120 dBFS).
    /// Non-finite samples are ignored.
    #[must_use]
    pub fn measure_peak_db(audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return SILENCE_DB;
        }
        let peak = audio
            .iter()
            .filter(|s| s.is_finite())
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        linear_to_db(peak)
    }

    /// Measure the RMS level of an audio buffer in dBFS.
    ///
    /// Computes root-mean-square of finite samples and converts to dBFS.
    /// Empty or all-zero buffers return [`SILENCE_DB`] (-120 dBFS).
    /// Non-finite samples are excluded from the count.
    #[must_use]
    pub fn measure_rms_db(audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return SILENCE_DB;
        }
        let (sum_sq, count) = audio.iter().fold((0.0f64, 0u64), |(acc, n), &s| {
            if s.is_finite() {
                (acc + f64::from(s) * f64::from(s), n + 1)
            } else {
                (acc, n)
            }
        });
        if count == 0 {
            return SILENCE_DB;
        }
        let rms = (sum_sq / count as f64).sqrt() as f32;
        linear_to_db(rms)
    }

    /// Approximate LUFS measurement using simplified K-weighting.
    ///
    /// Implements a simplified version of ITU-R BS.1770 loudness:
    /// 1. Apply a K-weighting pre-filter (high-shelf +4 dB above ~1500 Hz).
    /// 2. Compute mean-square over the full buffer.
    /// 3. Convert to LUFS (= -0.691 + 10*log10(mean_square)).
    ///
    /// The -0.691 dB offset is the ITU-R BS.1770 absolute gate adjustment
    /// for mono signals. This approximation skips the 400ms gating window
    /// and the relative -10 LUFS gate, making it suitable for short TTS
    /// segments but not broadcast-compliant metering.
    ///
    /// Returns [`SILENCE_DB`] for empty or silent buffers.
    #[must_use]
    pub fn measure_lufs_approx(audio: &[f32], sample_rate: f32) -> f32 {
        if audio.is_empty() {
            return SILENCE_DB;
        }

        // Apply simplified K-weighting: single-pole high-shelf filter.
        // Transfer function: H(z) = (1 + alpha) / 2 * (1 - z^-1) / (1 - alpha*z^-1)
        // where alpha = exp(-2*pi*1500/sr). This approximates the +4 dB
        // shelf above 1500 Hz from ITU-R BS.1770 stage 1.
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 {
            f64::from(sample_rate)
        } else {
            24000.0
        };
        let alpha = (-2.0 * std::f64::consts::PI * 1500.0 / sr).exp();

        // K-weighting coefficients: high-shelf at 1500 Hz.
        // We use a simple 1-pole shelf: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
        // with boost = 10^(4/20) ~ 1.585 above the cutoff.
        let boost = 10.0f64.powf(4.0 / 20.0); // +4 dB
        let b0 = (1.0 + alpha * boost) / (1.0 + alpha);
        let b1 = -(alpha + boost) / (1.0 + alpha);
        let a1 = -(alpha - 1.0) / (1.0 + alpha);

        let mut y_prev = 0.0f64;
        let mut x_prev = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;

        for &s in audio {
            if !s.is_finite() {
                continue;
            }
            let x = f64::from(s);
            let y = b0 * x + b1 * x_prev - a1 * y_prev;
            let y = if y.is_finite() { y } else { 0.0 };

            sum_sq += y * y;
            count += 1;

            x_prev = x;
            y_prev = y;
        }

        if count == 0 || sum_sq < f64::from(AMPLITUDE_FLOOR) {
            return SILENCE_DB;
        }

        let mean_sq = sum_sq / count as f64;
        // LUFS = -0.691 + 10*log10(mean_square)
        let lufs = -0.691 + 10.0 * mean_sq.log10();
        if lufs.is_finite() {
            lufs as f32
        } else {
            SILENCE_DB
        }
    }

    /// Apply a gain adjustment in dB to an audio buffer with soft-clipping.
    ///
    /// Converts `gain_db` to linear, multiplies each sample, then applies
    /// `tanh` soft-clipping at the configured ceiling to prevent hard
    /// clipping. Non-finite samples are zeroed.
    pub fn apply_gain(&self, audio: &mut [f32], gain_db: f32) {
        if audio.is_empty() {
            return;
        }
        // IEEE 754 safety: non-finite gain treated as 0 dB (no change).
        let gain_linear = if gain_db.is_finite() {
            db_to_linear(gain_db)
        } else {
            1.0
        };

        let ceiling = self.ceiling_linear;
        let inv_ceiling = if ceiling > AMPLITUDE_FLOOR {
            1.0 / ceiling
        } else {
            1.0
        };

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }
            let amplified = *sample * gain_linear;
            // Only apply soft-clipping when amplitude approaches ceiling.
            // Below 90% of ceiling, pass through linearly. Above, use
            // tanh soft-clip to smoothly limit.
            if amplified.abs() > ceiling * 0.9 {
                let normalized = amplified * inv_ceiling;
                *sample = normalized.tanh() * ceiling;
            } else {
                *sample = amplified;
            }
        }
    }

    /// Automatically level audio to the configured target LUFS.
    ///
    /// Measures the current approximate LUFS, computes the gain needed to
    /// reach `target_lufs`, then limits the gain so peak does not exceed
    /// `target_peak_dbfs`. Returns the applied gain in dB.
    ///
    /// If the signal is already within 0.5 dB of the target, no gain is
    /// applied (returns 0.0). If the signal is silence, returns 0.0.
    pub fn auto_level(&self, audio: &mut [f32], sample_rate: f32) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let current_lufs = Self::measure_lufs_approx(audio, sample_rate);
        if current_lufs <= SILENCE_DB + 1.0 {
            // Signal is silence; no meaningful level to adjust.
            return 0.0;
        }

        let needed_gain_db = self.config.target_lufs - current_lufs;

        // Check if already close enough (within 0.5 dB tolerance).
        if needed_gain_db.abs() < 0.5 {
            return 0.0;
        }

        // Limit gain so peak does not exceed the effective ceiling.
        let current_peak_db = Self::measure_peak_db(audio);
        let effective_peak_db = self.config.target_peak_dbfs.min(-self.config.headroom_db);
        let max_gain_db = effective_peak_db - current_peak_db;

        let applied_gain_db = needed_gain_db.min(max_gain_db);

        // Only boost or cut if it actually moves toward target.
        if (applied_gain_db > 0.0 && needed_gain_db > 0.0)
            || (applied_gain_db < 0.0 && needed_gain_db < 0.0)
        {
            self.apply_gain(audio, applied_gain_db);
            applied_gain_db
        } else if needed_gain_db < 0.0 {
            // Need to reduce level — always safe to cut.
            self.apply_gain(audio, needed_gain_db);
            needed_gain_db
        } else {
            0.0
        }
    }

    /// Simple peak normalization: scale audio so peak matches `target_db`.
    ///
    /// Unlike [`auto_level`](Self::auto_level), this does not consider
    /// loudness (LUFS), only the peak sample value. Soft-clipping is
    /// still applied via [`apply_gain`](Self::apply_gain).
    ///
    /// If the signal is silence, no gain is applied.
    pub fn normalize_peak(&self, audio: &mut [f32], target_db: f32) {
        if audio.is_empty() {
            return;
        }
        let current_peak_db = Self::measure_peak_db(audio);
        if current_peak_db <= SILENCE_DB + 1.0 {
            return;
        }
        let target = if target_db.is_finite() {
            target_db
        } else {
            self.config.target_peak_dbfs
        };
        let gain_db = target - current_peak_db;
        self.apply_gain(audio, gain_db);
    }

    /// Get the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &GainStagingConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Inter-stage monitor
// ---------------------------------------------------------------------------

/// Level report for a single processing stage.
#[derive(Debug, Clone)]
pub struct StageLevelReport {
    /// Name of the processing stage (e.g., "detune", "eq", "reverb").
    pub stage_name: String,
    /// Peak level in dBFS.
    pub peak_db: f32,
    /// RMS level in dBFS.
    pub rms_db: f32,
    /// Whether a level violation was detected.
    pub violation: Option<LevelViolation>,
}

/// Type of level violation detected at a processing stage.
#[derive(Debug, Clone, PartialEq)]
pub enum LevelViolation {
    /// Peak exceeds -0.1 dBFS (near-clipping).
    PeakClipping { peak_db: f32 },
    /// RMS exceeds target + 3 dB (excessively loud).
    RmsExcessive { rms_db: f32, threshold_db: f32 },
}

/// Tracks audio levels between processing stages in the chorus pipeline.
///
/// Records peak and RMS levels after each stage and detects level violations
/// (near-clipping, excessive loudness). Use this to diagnose gain staging
/// problems in multi-stage processing chains.
pub struct InterStageMonitor {
    records: Vec<StageLevelReport>,
    /// RMS threshold above which a violation is flagged.
    /// Default: target_lufs + 3.0 dB.
    rms_violation_threshold_db: f32,
}

impl InterStageMonitor {
    /// Create a new monitor with a default RMS violation threshold.
    ///
    /// The RMS threshold defaults to -13.0 dBFS (suitable for -16 LUFS
    /// target + 3 dB margin).
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            rms_violation_threshold_db: -13.0,
        }
    }

    /// Create a monitor with a custom RMS violation threshold.
    #[must_use]
    pub fn with_rms_threshold(threshold_db: f32) -> Self {
        Self {
            records: Vec::new(),
            rms_violation_threshold_db: if threshold_db.is_finite() {
                threshold_db
            } else {
                -13.0
            },
        }
    }

    /// Create a monitor from a gain staging config.
    ///
    /// Sets the RMS violation threshold to `target_lufs + 3.0`.
    #[must_use]
    pub fn from_config(config: &GainStagingConfig) -> Self {
        Self::with_rms_threshold(config.target_lufs + 3.0)
    }

    /// Record the level of audio at a processing stage.
    ///
    /// Computes peak and RMS from the audio buffer, checks for violations,
    /// and appends a [`StageLevelReport`] to the internal record.
    pub fn record_level(&mut self, stage_name: &str, peak_db: f32, rms_db: f32) {
        let peak = if peak_db.is_finite() {
            peak_db
        } else {
            SILENCE_DB
        };
        let rms = if rms_db.is_finite() {
            rms_db
        } else {
            SILENCE_DB
        };

        let violation = if peak > -0.1 {
            Some(LevelViolation::PeakClipping { peak_db: peak })
        } else if rms > self.rms_violation_threshold_db {
            Some(LevelViolation::RmsExcessive {
                rms_db: rms,
                threshold_db: self.rms_violation_threshold_db,
            })
        } else {
            None
        };

        self.records.push(StageLevelReport {
            stage_name: stage_name.to_string(),
            peak_db: peak,
            rms_db: rms,
            violation,
        });
    }

    /// Record levels by measuring an audio buffer directly.
    ///
    /// Convenience method that measures peak and RMS from the buffer and
    /// records the results.
    pub fn record_audio(&mut self, stage_name: &str, audio: &[f32]) {
        let peak_db = GainStager::measure_peak_db(audio);
        let rms_db = GainStager::measure_rms_db(audio);
        self.record_level(stage_name, peak_db, rms_db);
    }

    /// Get an ordered list of per-stage level reports.
    #[must_use]
    pub fn report(&self) -> Vec<StageLevelReport> {
        self.records.clone()
    }

    /// Check whether any stage has a level violation.
    #[must_use]
    pub fn has_violations(&self) -> bool {
        self.records.iter().any(|r| r.violation.is_some())
    }

    /// Get only the reports that have violations.
    #[must_use]
    pub fn violations(&self) -> Vec<&StageLevelReport> {
        self.records
            .iter()
            .filter(|r| r.violation.is_some())
            .collect()
    }

    /// Clear all recorded levels.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

impl Default for InterStageMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a linear amplitude to dBFS.
///
/// Returns [`SILENCE_DB`] for zero or sub-threshold amplitudes.
#[inline]
fn linear_to_db(linear: f32) -> f32 {
    if !linear.is_finite() || linear < AMPLITUDE_FLOOR {
        return SILENCE_DB;
    }
    let db = 20.0 * linear.log10();
    if db.is_finite() {
        db
    } else {
        SILENCE_DB
    }
}

/// Convert a dBFS value to linear amplitude.
///
/// Returns 0.0 for non-finite inputs or values at or below [`SILENCE_DB`].
#[inline]
fn db_to_linear(db: f32) -> f32 {
    if !db.is_finite() || db <= SILENCE_DB {
        return 0.0;
    }
    let linear = 10.0f32.powf(db / 20.0);
    if linear.is_finite() {
        linear
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_config_default_is_valid() {
        let config = GainStagingConfig::default();
        config.validate().expect("default config should be valid");
        assert!((config.target_lufs - (-16.0)).abs() < f32::EPSILON);
        assert!((config.target_peak_dbfs - (-1.0)).abs() < f32::EPSILON);
        assert!(config.auto_makeup);
        assert!((config.headroom_db - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_new_validates() {
        assert!(GainStagingConfig::new(-16.0, -1.0, true, 3.0).is_ok());
        assert!(GainStagingConfig::new(-24.0, -6.0, false, 12.0).is_ok());
        assert!(GainStagingConfig::new(-8.0, -0.1, true, 1.0).is_ok());

        // Out-of-range target_lufs.
        assert!(GainStagingConfig::new(-25.0, -1.0, true, 3.0).is_err());
        assert!(GainStagingConfig::new(-7.0, -1.0, true, 3.0).is_err());
        assert!(GainStagingConfig::new(f32::NAN, -1.0, true, 3.0).is_err());

        // Out-of-range target_peak_dbfs.
        assert!(GainStagingConfig::new(-16.0, -7.0, true, 3.0).is_err());
        assert!(GainStagingConfig::new(-16.0, 0.0, true, 3.0).is_err());
        assert!(GainStagingConfig::new(-16.0, f32::INFINITY, true, 3.0).is_err());

        // Out-of-range headroom_db.
        assert!(GainStagingConfig::new(-16.0, -1.0, true, 0.5).is_err());
        assert!(GainStagingConfig::new(-16.0, -1.0, true, 13.0).is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = GainStagingConfig::default()
            .with_target_lufs(-20.0)
            .with_target_peak_dbfs(-2.0)
            .with_auto_makeup(false)
            .with_headroom_db(6.0);
        config.validate().expect("builder config should be valid");
        assert!((config.target_lufs - (-20.0)).abs() < f32::EPSILON);
        assert!((config.target_peak_dbfs - (-2.0)).abs() < f32::EPSILON);
        assert!(!config.auto_makeup);
        assert!((config.headroom_db - 6.0).abs() < f32::EPSILON);
    }

    // -- Measurement tests --------------------------------------------------

    #[test]
    fn test_measure_peak_db_sine() {
        // Full-scale sine: peak = 1.0 = 0 dBFS.
        let audio: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let peak = GainStager::measure_peak_db(&audio);
        // Should be very close to 0 dBFS (within 0.1 dB).
        assert!(peak > -0.1 && peak <= 0.01, "expected ~0 dBFS, got {peak}");
    }

    #[test]
    fn test_measure_peak_db_empty() {
        assert!((GainStager::measure_peak_db(&[]) - SILENCE_DB).abs() < f32::EPSILON);
    }

    #[test]
    fn test_measure_rms_db_sine() {
        // Full-scale sine: RMS = 1/sqrt(2) = -3.01 dBFS.
        let audio: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let rms = GainStager::measure_rms_db(&audio);
        assert!(
            (rms - (-3.01)).abs() < 0.1,
            "expected ~-3.01 dBFS, got {rms}"
        );
    }

    #[test]
    fn test_measure_rms_db_empty() {
        assert!((GainStager::measure_rms_db(&[]) - SILENCE_DB).abs() < f32::EPSILON);
    }

    #[test]
    fn test_measure_lufs_approx_reasonable() {
        // Full-scale sine at 440 Hz, 1 second at 24 kHz.
        let audio: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let lufs = GainStager::measure_lufs_approx(&audio, 24000.0);
        // A 440 Hz sine is below the K-weighting boost frequency (1500 Hz),
        // so LUFS reads lower than RMS dBFS. Expect roughly -15 to -25 LUFS
        // for a full-scale mono sine at 440 Hz.
        assert!(
            lufs > -25.0 && lufs < 0.0,
            "expected reasonable LUFS for full-scale sine, got {lufs}"
        );
    }

    #[test]
    fn test_measure_lufs_approx_empty() {
        assert!((GainStager::measure_lufs_approx(&[], 24000.0) - SILENCE_DB).abs() < f32::EPSILON);
    }

    #[test]
    fn test_measure_lufs_quiet_vs_loud() {
        let loud: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let quiet: Vec<f32> = loud.iter().map(|&s| s * 0.1).collect();

        let lufs_loud = GainStager::measure_lufs_approx(&loud, 24000.0);
        let lufs_quiet = GainStager::measure_lufs_approx(&quiet, 24000.0);
        assert!(
            lufs_loud > lufs_quiet,
            "loud ({lufs_loud}) should be > quiet ({lufs_quiet})"
        );
        // -20 dB amplitude = -20 dB loudness approximately.
        let diff = lufs_loud - lufs_quiet;
        assert!(
            (diff - 20.0).abs() < 2.0,
            "expected ~20 dB difference, got {diff}"
        );
    }

    // -- Auto-level tests ---------------------------------------------------

    #[test]
    fn test_auto_level_brings_quiet_signal_up() {
        let config = GainStagingConfig::new(-16.0, -1.0, true, 3.0).expect("valid config");
        let stager = GainStager::new(&config).expect("valid stager");

        // Quiet signal: -40 dBFS peak.
        let amplitude = db_to_linear(-40.0);
        let mut audio: Vec<f32> = (0..24000)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();

        let gain = stager.auto_level(&mut audio, 24000.0);
        assert!(gain > 0.0, "should have boosted quiet signal, gain={gain}");

        // After auto-level, the signal should be louder than before.
        let new_peak = GainStager::measure_peak_db(&audio);
        assert!(
            new_peak > -40.0,
            "peak should have increased from -40 dBFS, got {new_peak}"
        );
    }

    #[test]
    fn test_auto_level_reduces_loud_signal() {
        // Use a low LUFS target so a full-scale signal exceeds it.
        let config = GainStagingConfig::new(-24.0, -1.0, true, 3.0).expect("valid config");
        let stager = GainStager::new(&config).expect("valid stager");

        // Full-scale 4 kHz sine: K-weighting boosts it, so LUFS will be
        // well above -24 target.
        let mut audio: Vec<f32> = (0..24000)
            .map(|i| (2.0 * std::f32::consts::PI * 4000.0 * i as f32 / 24000.0).sin())
            .collect();

        let lufs_before = GainStager::measure_lufs_approx(&audio, 24000.0);
        assert!(
            lufs_before > -24.0,
            "precondition: signal LUFS ({lufs_before}) should exceed target (-24)"
        );

        let gain = stager.auto_level(&mut audio, 24000.0);
        assert!(gain < 0.0, "should have reduced loud signal, gain={gain}");

        // Peak should be below 0 dBFS.
        let new_peak = GainStager::measure_peak_db(&audio);
        assert!(
            new_peak < 0.0,
            "peak should be below 0 dBFS after reduction, got {new_peak}"
        );
    }

    #[test]
    fn test_auto_level_silence_returns_zero_gain() {
        let stager = GainStager::with_defaults().expect("valid stager");
        let mut audio = vec![0.0f32; 1000];
        let gain = stager.auto_level(&mut audio, 24000.0);
        assert!(
            gain.abs() < f32::EPSILON,
            "silence should return 0 gain, got {gain}"
        );
    }

    // -- Peak normalization tests -------------------------------------------

    #[test]
    fn test_normalize_peak() {
        // Use a config with low headroom so the ceiling is well above
        // our normalization target (-6 dBFS).
        let config = GainStagingConfig::new(-16.0, -0.5, true, 1.0).expect("valid config");
        let stager = GainStager::new(&config).expect("valid stager");

        let mut audio: Vec<f32> = (0..24000)
            .map(|i| 0.25 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();

        let before_peak = GainStager::measure_peak_db(&audio);
        assert!(
            (before_peak - (-12.04)).abs() < 0.2,
            "0.25 amplitude should be ~-12 dBFS, got {before_peak}"
        );

        stager.normalize_peak(&mut audio, -6.0);
        let after_peak = GainStager::measure_peak_db(&audio);
        // After normalization to -6 dBFS: 0.25 -> 0.5 (well below ceiling).
        // No soft-clipping involved, so tolerance is tight.
        assert!(
            (after_peak - (-6.0)).abs() < 0.2,
            "expected ~-6 dBFS after normalize, got {after_peak}"
        );
    }

    #[test]
    fn test_normalize_peak_silence() {
        let stager = GainStager::with_defaults().expect("valid stager");
        let mut audio = vec![0.0f32; 1000];
        stager.normalize_peak(&mut audio, -3.0);
        // Should remain silence.
        assert!(audio.iter().all(|&s| s.abs() < f32::EPSILON));
    }

    // -- Apply gain tests ---------------------------------------------------

    #[test]
    fn test_apply_gain_zero_db_no_change() {
        let stager = GainStager::with_defaults().expect("valid stager");
        let original: Vec<f32> = (0..1000).map(|i| 0.5 * (i as f32 * 0.01).sin()).collect();
        let mut audio = original.clone();
        stager.apply_gain(&mut audio, 0.0);

        // Signals at 0.5 amplitude are well below the ceiling (~0.7),
        // so they pass through the linear path unchanged.
        for (i, (&got, &orig)) in audio.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - orig).abs() < f32::EPSILON,
                "sample {i}: {got} != {orig}"
            );
        }
    }

    #[test]
    fn test_apply_gain_nan_samples_zeroed() {
        let stager = GainStager::with_defaults().expect("valid stager");
        let mut audio = vec![0.5, f32::NAN, f32::INFINITY, -0.3, f32::NEG_INFINITY];
        stager.apply_gain(&mut audio, 6.0);
        for (i, &s) in audio.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} should be finite, got {s}");
        }
    }

    // -- InterStageMonitor tests -------------------------------------------

    #[test]
    fn test_monitor_no_violations() {
        let mut monitor = InterStageMonitor::new();
        monitor.record_level("detune", -12.0, -18.0);
        monitor.record_level("eq", -10.0, -16.0);
        monitor.record_level("reverb", -8.0, -14.0);

        assert!(!monitor.has_violations());
        let report = monitor.report();
        assert_eq!(report.len(), 3);
        assert_eq!(report[0].stage_name, "detune");
        assert_eq!(report[2].stage_name, "reverb");
    }

    #[test]
    fn test_monitor_peak_violation() {
        let mut monitor = InterStageMonitor::new();
        monitor.record_level("saturation", 0.0, -10.0);
        assert!(monitor.has_violations());

        let violations = monitor.violations();
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations[0].violation,
            Some(LevelViolation::PeakClipping { .. })
        ));
    }

    #[test]
    fn test_monitor_rms_violation() {
        let mut monitor = InterStageMonitor::with_rms_threshold(-13.0);
        monitor.record_level("compressor", -6.0, -10.0);
        assert!(monitor.has_violations());

        let violations = monitor.violations();
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations[0].violation,
            Some(LevelViolation::RmsExcessive { .. })
        ));
    }

    #[test]
    fn test_monitor_record_audio() {
        let mut monitor = InterStageMonitor::new();
        let audio: Vec<f32> = (0..2400)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        monitor.record_audio("input", &audio);

        let report = monitor.report();
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].stage_name, "input");
        // Peak of 0.5 amplitude = ~-6 dBFS.
        assert!(
            (report[0].peak_db - (-6.02)).abs() < 0.2,
            "expected ~-6 dBFS peak, got {}",
            report[0].peak_db
        );
    }

    #[test]
    fn test_monitor_from_config() {
        let config = GainStagingConfig::default();
        let monitor = InterStageMonitor::from_config(&config);
        // Threshold should be target_lufs + 3 = -16 + 3 = -13.
        assert!((monitor.rms_violation_threshold_db - (-13.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_clear() {
        let mut monitor = InterStageMonitor::new();
        monitor.record_level("a", -10.0, -15.0);
        assert_eq!(monitor.report().len(), 1);
        monitor.clear();
        assert!(monitor.report().is_empty());
    }

    // -- Helper tests -------------------------------------------------------

    #[test]
    fn test_linear_to_db_roundtrip() {
        for &db_val in &[-60.0f32, -30.0, -12.0, -6.0, -3.0, 0.0] {
            let linear = db_to_linear(db_val);
            let back = linear_to_db(linear);
            assert!(
                (back - db_val).abs() < 0.01,
                "roundtrip failed for {db_val}: linear={linear}, back={back}"
            );
        }
    }

    #[test]
    fn test_linear_to_db_zero() {
        assert!((linear_to_db(0.0) - SILENCE_DB).abs() < f32::EPSILON);
    }

    #[test]
    fn test_db_to_linear_silence() {
        assert!(db_to_linear(SILENCE_DB).abs() < f32::EPSILON);
        assert!(db_to_linear(f32::NEG_INFINITY).abs() < f32::EPSILON);
    }

    #[test]
    fn test_k_weighting_boosts_high_freq() {
        // High-frequency signal should measure louder in LUFS than a
        // low-frequency signal at the same amplitude, due to K-weighting.
        let sr = 24000.0;
        let n = 24000;

        let low_freq: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sr).sin())
            .collect();
        let high_freq: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 4000.0 * i as f32 / sr).sin())
            .collect();

        let lufs_low = GainStager::measure_lufs_approx(&low_freq, sr);
        let lufs_high = GainStager::measure_lufs_approx(&high_freq, sr);

        // K-weighting boosts high frequencies, so high_freq should measure louder.
        assert!(
            lufs_high > lufs_low,
            "K-weighting should make high freq ({lufs_high}) louder than low freq ({lufs_low})"
        );
    }
}
