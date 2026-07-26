// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Stereo correlation and phase analysis for Kokoro chorus output.
//!
//! Monitors stereo signals for phase issues, mono compatibility, and
//! correlation problems. Optionally corrects detected issues in real-time
//! by adjusting mid/side balance and enforcing bass mono coherence.
//!
//! # Why this matters
//!
//! Chorus effects generate multiple voices panned across the stereo field.
//! Poorly correlated stereo content causes:
//! - **Woofer cancellation:** Out-of-phase bass frequencies cancel on mono
//!   playback systems (phone speakers, Bluetooth, broadcast).
//! - **Comb-filtering artifacts:** Phase differences create notch patterns
//!   when summed to mono.
//! - **Perceived loudness loss:** Anti-correlated content reduces perceived
//!   loudness on mono and narrow-stereo playback.
//!
//! # Metrics computed
//!
//! - **Correlation coefficient:** Pearson correlation of L and R channels.
//!   +1 = identical (mono), 0 = uncorrelated, -1 = perfectly out of phase.
//! - **Mid/side levels:** RMS of mid (L+R)/2 and side (L-R)/2 in dB.
//! - **Phase offset:** Average instantaneous phase difference in degrees.
//! - **Bass correlation:** Correlation of low-frequency content only.
//! - **Mono compatible:** Whether the signal will survive mono downmix
//!   without significant cancellation.
//!
//! # Correction
//!
//! When `enable_correction` is true and correlation drops below the
//! configured threshold, the processor gradually narrows the stereo image
//! in the problematic region. Bass mono enforcement forces frequencies
//! below `bass_mono_below_hz` toward mono to prevent woofer cancellation.
//!
//! Part of #4264, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for stereo correlation analysis and correction.
///
/// Controls what metrics are computed and whether automatic correction
/// is applied when correlation problems are detected.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StereoAnalysisConfig {
    /// Enable metric computation. When false, `analyze` returns default
    /// metrics without scanning the audio.
    pub enable_monitoring: bool,

    /// Enable automatic correction of detected phase issues.
    /// Requires `enable_monitoring` to be true.
    pub enable_correction: bool,

    /// Minimum acceptable correlation coefficient before correction
    /// engages. Range: -1.0 to 1.0. Default: 0.0 (correction triggers
    /// when channels become anti-correlated).
    pub min_correlation: f32,

    /// Target correlation to pull toward during correction.
    /// Range: 0.0 to 1.0. Default: 0.3.
    pub target_correlation: f32,

    /// How aggressively to apply correction. 0.0 = no effect, 1.0 = full
    /// snap toward mono. Default: 0.5.
    pub correction_strength: f32,

    /// Frequencies below this threshold are forced mono to prevent
    /// woofer cancellation. Set to 0.0 to disable bass mono enforcement.
    /// Default: 120.0 Hz.
    pub bass_mono_below_hz: f32,

    /// Audio sample rate in Hz. Required for bass filter computation.
    /// Default: 24000.0 (Kokoro native rate).
    pub sample_rate: f32,
}

impl StereoAnalysisConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether monitoring is enabled.
    #[must_use]
    pub fn with_monitoring(mut self, enable: bool) -> Self {
        self.enable_monitoring = enable;
        self
    }

    /// Set whether correction is enabled.
    #[must_use]
    pub fn with_correction(mut self, enable: bool) -> Self {
        self.enable_correction = enable;
        self
    }

    /// Set minimum correlation threshold.
    #[must_use]
    pub fn with_min_correlation(mut self, min: f32) -> Self {
        self.min_correlation = min;
        self
    }

    /// Set target correlation for correction.
    #[must_use]
    pub fn with_target_correlation(mut self, target: f32) -> Self {
        self.target_correlation = target;
        self
    }

    /// Set correction strength.
    #[must_use]
    pub fn with_correction_strength(mut self, strength: f32) -> Self {
        self.correction_strength = strength;
        self
    }

    /// Set bass mono enforcement frequency.
    #[must_use]
    pub fn with_bass_mono_below_hz(mut self, freq: f32) -> Self {
        self.bass_mono_below_hz = freq;
        self
    }

    /// Set audio sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, rate: f32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Validate all parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range
    /// or non-finite.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.min_correlation.is_finite() || !(-1.0..=1.0).contains(&self.min_correlation) {
            return Err(KokoroError::InvalidConfig {
                field: "min_correlation",
                reason: format!(
                    "min_correlation = {}: must be finite and in [-1.0, 1.0]",
                    self.min_correlation,
                ),
            });
        }
        if !self.target_correlation.is_finite() || !(0.0..=1.0).contains(&self.target_correlation) {
            return Err(KokoroError::InvalidConfig {
                field: "target_correlation",
                reason: format!(
                    "target_correlation = {}: must be finite and in [0.0, 1.0]",
                    self.target_correlation,
                ),
            });
        }
        if !self.correction_strength.is_finite() || !(0.0..=1.0).contains(&self.correction_strength)
        {
            return Err(KokoroError::InvalidConfig {
                field: "correction_strength",
                reason: format!(
                    "correction_strength = {}: must be finite and in [0.0, 1.0]",
                    self.correction_strength,
                ),
            });
        }
        if !self.bass_mono_below_hz.is_finite() || self.bass_mono_below_hz < 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "bass_mono_below_hz",
                reason: format!(
                    "bass_mono_below_hz = {}: must be finite and >= 0.0",
                    self.bass_mono_below_hz,
                ),
            });
        }
        if !self.sample_rate.is_finite() || self.sample_rate < 1000.0 || self.sample_rate > 192000.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!(
                    "sample_rate = {}: must be finite and in [1000, 192000]",
                    self.sample_rate,
                ),
            });
        }
        if self.min_correlation > self.target_correlation {
            return Err(KokoroError::InvalidConfig {
                field: "min_correlation",
                reason: format!(
                    "min_correlation ({}) must be <= target_correlation ({})",
                    self.min_correlation, self.target_correlation,
                ),
            });
        }
        Ok(())
    }
}

impl Default for StereoAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            enable_correction: false,
            min_correlation: 0.0,
            target_correlation: 0.3,
            correction_strength: 0.5,
            bass_mono_below_hz: 120.0,
            sample_rate: 24000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Stereo analysis metrics for a block of audio.
///
/// All level measurements are in dB relative to full-scale (dBFS).
/// A value of 0.0 dBFS corresponds to a signal at digital maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StereoMetrics {
    /// Pearson correlation coefficient between L and R.
    /// +1.0 = identical (mono), 0.0 = uncorrelated, -1.0 = out of phase.
    pub correlation: f32,

    /// RMS level of the mid (L+R)/2 signal in dBFS.
    pub mid_level_db: f32,

    /// RMS level of the side (L-R)/2 signal in dBFS.
    pub side_level_db: f32,

    /// Average instantaneous phase offset in degrees (0..180).
    pub phase_offset_deg: f32,

    /// Correlation of low-frequency content only (below bass threshold).
    pub bass_correlation: f32,

    /// Whether the signal is mono-compatible (correlation > min_threshold
    /// and bass correlation > 0.5).
    pub mono_compatible: bool,
}

impl Default for StereoMetrics {
    fn default() -> Self {
        Self {
            correlation: 1.0,
            mid_level_db: f32::NEG_INFINITY,
            side_level_db: f32::NEG_INFINITY,
            phase_offset_deg: 0.0,
            bass_correlation: 1.0,
            mono_compatible: true,
        }
    }
}

// ---------------------------------------------------------------------------
// One-pole LP filter for bass extraction
// ---------------------------------------------------------------------------

/// Simple one-pole lowpass filter for bass frequency extraction.
#[derive(Debug, Clone)]
struct BassFilter {
    /// Filter coefficient.
    g: f32,
    /// Filter state for left channel.
    z1_l: f32,
    /// Filter state for right channel.
    z1_r: f32,
}

impl BassFilter {
    /// Create a new bass filter for the given cutoff and sample rate.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let g = if cutoff_hz <= 0.0 || sample_rate <= 0.0 {
            0.0
        } else {
            (1.0 - (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate).exp())
                .clamp(0.001, 0.999)
        };
        Self {
            g,
            z1_l: 0.0,
            z1_r: 0.0,
        }
    }

    /// Extract lowpass component of left and right channels.
    #[inline]
    fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        if !left.is_finite() || !right.is_finite() {
            self.z1_l = 0.0;
            self.z1_r = 0.0;
            return (0.0, 0.0);
        }
        self.z1_l += self.g * (left - self.z1_l);
        self.z1_r += self.g * (right - self.z1_r);
        // Flush denormals.
        if self.z1_l.abs() < 1e-20 {
            self.z1_l = 0.0;
        }
        if self.z1_r.abs() < 1e-20 {
            self.z1_r = 0.0;
        }
        (self.z1_l, self.z1_r)
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.z1_l = 0.0;
        self.z1_r = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stereo correlation analyzer and optional corrector.
///
/// Computes per-block stereo metrics and optionally fixes phase issues
/// by adjusting mid/side balance when correlation drops below threshold.
///
/// # Usage
///
/// ```rust,no_run
/// use nn_models::kokoro_chorus_stereo_analysis::*;
///
/// let config = StereoAnalysisConfig::new()
///     .with_correction(true)
///     .with_bass_mono_below_hz(120.0);
/// let mut analyzer = StereoAnalyzer::new(config).unwrap();
///
/// let mut left = vec![0.0f32; 1024];
/// let mut right = vec![0.0f32; 1024];
/// // ... fill with chorus output ...
/// let metrics = analyzer.process(&mut left, &mut right);
/// assert!(metrics.mono_compatible);
/// ```
pub struct StereoAnalyzer {
    /// Configuration.
    config: StereoAnalysisConfig,
    /// Bass extraction filter for bass correlation measurement.
    bass_filter_analysis: BassFilter,
    /// Bass extraction filter for bass mono enforcement (correction path).
    bass_filter_correct: BassFilter,
}

impl StereoAnalyzer {
    /// Create a new stereo analyzer with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails.
    pub fn new(config: StereoAnalysisConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let bass_filter_analysis = BassFilter::new(config.bass_mono_below_hz, config.sample_rate);
        let bass_filter_correct = BassFilter::new(config.bass_mono_below_hz, config.sample_rate);
        Ok(Self {
            config,
            bass_filter_analysis,
            bass_filter_correct,
        })
    }

    /// Analyze stereo audio without modifying it.
    ///
    /// Returns metrics describing the stereo correlation, phase offset,
    /// mid/side balance, and mono compatibility.
    #[must_use]
    pub fn analyze(&self, left: &[f32], right: &[f32]) -> StereoMetrics {
        if !self.config.enable_monitoring {
            return StereoMetrics::default();
        }

        let len = left.len().min(right.len());
        if len == 0 {
            return StereoMetrics::default();
        }

        // Use a temporary bass filter clone for analysis (immutable self).
        let mut bass_filt =
            BassFilter::new(self.config.bass_mono_below_hz, self.config.sample_rate);

        compute_metrics(left, right, len, &mut bass_filt, &self.config)
    }

    /// Correct phase issues in stereo audio in-place.
    ///
    /// When correlation is below `min_correlation`, gradually pulls the
    /// stereo image toward mono by the configured `correction_strength`.
    /// Enforces bass mono below `bass_mono_below_hz`.
    pub fn correct(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.config.enable_correction {
            return;
        }

        let len = left.len().min(right.len());
        if len == 0 {
            return;
        }

        // Compute running correlation for adaptive correction.
        let corr = pearson_correlation(left, right, len);

        // Apply broadband mid/side correction if correlation is low.
        if corr < self.config.min_correlation {
            let amount = self.config.correction_strength;
            apply_correlation_correction(left, right, len, amount);
        }

        // Enforce bass mono.
        if self.config.bass_mono_below_hz > 0.0 {
            apply_bass_mono(left, right, len, &mut self.bass_filter_correct);
        }
    }

    /// Analyze and correct in one pass: returns metrics, then applies fixes.
    ///
    /// Equivalent to calling `analyze` followed by `correct`, but measures
    /// metrics on the original signal before correction modifies it.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) -> StereoMetrics {
        let metrics = if self.config.enable_monitoring {
            let len = left.len().min(right.len());
            if len == 0 {
                StereoMetrics::default()
            } else {
                compute_metrics(
                    left,
                    right,
                    len,
                    &mut self.bass_filter_analysis,
                    &self.config,
                )
            }
        } else {
            StereoMetrics::default()
        };

        self.correct(left, right);
        metrics
    }

    /// Reset all internal filter state.
    ///
    /// Call between non-contiguous audio segments to avoid filter ringing.
    pub fn reset(&mut self) {
        self.bass_filter_analysis.reset();
        self.bass_filter_correct.reset();
    }

    /// Get a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &StereoAnalysisConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute Pearson correlation coefficient between two signals.
fn pearson_correlation(left: &[f32], right: &[f32], len: usize) -> f32 {
    if len == 0 {
        return 1.0;
    }

    let mut sum_l = 0.0_f64;
    let mut sum_r = 0.0_f64;
    let mut sum_ll = 0.0_f64;
    let mut sum_rr = 0.0_f64;
    let mut sum_lr = 0.0_f64;
    let mut count = 0u64;

    for i in 0..len {
        let l = f64::from(left[i]);
        let r = f64::from(right[i]);
        if !left[i].is_finite() || !right[i].is_finite() {
            continue;
        }
        sum_l += l;
        sum_r += r;
        sum_ll += l * l;
        sum_rr += r * r;
        sum_lr += l * r;
        count += 1;
    }

    if count == 0 {
        return 1.0;
    }

    let n = count as f64;
    let numerator = n * sum_lr - sum_l * sum_r;
    let denom_l = (n * sum_ll - sum_l * sum_l).max(0.0).sqrt();
    let denom_r = (n * sum_rr - sum_r * sum_r).max(0.0).sqrt();
    let denom = denom_l * denom_r;

    if denom < 1e-20 {
        // Both channels are constant (e.g. silence) -> perfectly correlated.
        return 1.0;
    }

    (numerator / denom).clamp(-1.0, 1.0) as f32
}

/// Compute full stereo metrics for a block.
fn compute_metrics(
    left: &[f32],
    right: &[f32],
    len: usize,
    bass_filter: &mut BassFilter,
    config: &StereoAnalysisConfig,
) -> StereoMetrics {
    let correlation = pearson_correlation(left, right, len);

    // Mid/side RMS computation.
    let mut mid_energy = 0.0_f64;
    let mut side_energy = 0.0_f64;
    let mut phase_sum = 0.0_f64;
    let mut phase_count = 0u64;

    // Bass correlation accumulators.
    let mut bass_sum_lr = 0.0_f64;
    let mut bass_sum_ll = 0.0_f64;
    let mut bass_sum_rr = 0.0_f64;

    for i in 0..len {
        let l = left[i];
        let r = right[i];

        if !l.is_finite() || !r.is_finite() {
            continue;
        }

        let ld = f64::from(l);
        let rd = f64::from(r);

        let mid = (ld + rd) * 0.5;
        let side = (ld - rd) * 0.5;
        mid_energy += mid * mid;
        side_energy += side * side;

        // Instantaneous phase offset: arccos of normalized dot product.
        let mag_l = ld.abs();
        let mag_r = rd.abs();
        if mag_l > 1e-10 && mag_r > 1e-10 {
            let cos_theta = (ld * rd) / (mag_l * mag_r);
            let angle = cos_theta.clamp(-1.0, 1.0).acos();
            phase_sum += angle;
            phase_count += 1;
        }

        // Bass correlation via lowpass filter.
        let (bass_l, bass_r) = bass_filter.process(l, r);
        let bl = f64::from(bass_l);
        let br = f64::from(bass_r);
        bass_sum_lr += bl * br;
        bass_sum_ll += bl * bl;
        bass_sum_rr += br * br;
    }

    let n = len as f64;
    let mid_rms = (mid_energy / n).sqrt();
    let side_rms = (side_energy / n).sqrt();

    let mid_level_db = rms_to_db(mid_rms as f32);
    let side_level_db = rms_to_db(side_rms as f32);

    let phase_offset_deg = if phase_count > 0 {
        ((phase_sum / phase_count as f64) * (180.0 / std::f64::consts::PI)) as f32
    } else {
        0.0
    };

    let bass_denom = (bass_sum_ll * bass_sum_rr).sqrt();
    let bass_correlation = if bass_denom > 1e-20 {
        (bass_sum_lr / bass_denom).clamp(-1.0, 1.0) as f32
    } else {
        1.0
    };

    let mono_compatible = correlation > config.min_correlation && bass_correlation > 0.5;

    StereoMetrics {
        correlation,
        mid_level_db,
        side_level_db,
        phase_offset_deg,
        bass_correlation,
        mono_compatible,
    }
}

/// Convert RMS amplitude to dBFS.
fn rms_to_db(rms: f32) -> f32 {
    if !rms.is_finite() || rms <= 0.0 {
        return f32::NEG_INFINITY;
    }
    20.0 * rms.log10()
}

/// Apply mid/side correction to pull stereo image toward mono.
///
/// `amount` controls how much side content is reduced: 0.0 = no change,
/// 1.0 = full collapse to mono.
fn apply_correlation_correction(left: &mut [f32], right: &mut [f32], len: usize, amount: f32) {
    let side_scale = 1.0 - amount;
    for i in 0..len {
        let l = left[i];
        let r = right[i];
        if !l.is_finite() || !r.is_finite() {
            left[i] = 0.0;
            right[i] = 0.0;
            continue;
        }
        let mid = (l + r) * 0.5;
        let side = (l - r) * 0.5 * side_scale;
        left[i] = mid + side;
        right[i] = mid - side;
    }
}

/// Enforce bass mono: replace low-frequency content with its mono sum.
fn apply_bass_mono(left: &mut [f32], right: &mut [f32], len: usize, bass_filter: &mut BassFilter) {
    for i in 0..len {
        let l = left[i];
        let r = right[i];

        if !l.is_finite() || !r.is_finite() {
            left[i] = 0.0;
            right[i] = 0.0;
            continue;
        }

        // Extract bass components.
        let (bass_l, bass_r) = bass_filter.process(l, r);

        // High-frequency residuals (pass through unchanged).
        let high_l = l - bass_l;
        let high_r = r - bass_r;

        // Sum bass to mono.
        let bass_mono = (bass_l + bass_r) * 0.5;

        left[i] = bass_mono + high_l;
        right[i] = bass_mono + high_r;

        // Clamp non-finite results.
        if !left[i].is_finite() {
            left[i] = 0.0;
        }
        if !right[i].is_finite() {
            right[i] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (extracted to separate file per 500-line rule)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_stereo_analysis_tests.rs"]
mod tests;
