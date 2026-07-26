// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Stereo image optimizer for the Kokoro chorus mixed output.
//!
//! Analyzes and optimizes the stereo image of a mixed chorus for maximum
//! perceived width, clarity, and mono compatibility. The optimizer works
//! in mid/side domain:
//!
//! - **Mid** = (L + R) / 2 -- the mono-compatible center content.
//! - **Side** = (L - R) / 2 -- the stereo-only width content.
//!
//! # Correlation monitoring
//!
//! L/R correlation is tracked with an exponential moving average. When
//! correlation drops below `min_correlation`, the optimizer automatically
//! narrows the stereo image by attenuating the side signal until
//! correlation recovers. This prevents phase cancellation on mono
//! playback systems (phone speakers, Bluetooth, broadcast).
//!
//! # Bass mono summing
//!
//! Frequencies below `bass_mono_freq_hz` (default 200 Hz) are forced to
//! mono by zeroing the side component of the low-frequency band. This
//! prevents woofer cancellation and tightens the low end on all playback
//! systems.
//!
//! # Width smoothing
//!
//! The effective width parameter is smoothed over `width_smoothing_ms` to
//! avoid audible pumping when the optimizer reacts to transient
//! correlation drops.
//!
//! References:
//! - Eargle, "Handbook of Recording Engineering," 4th ed., 2005. Ch. 14
//!   (stereo imaging, mid/side processing).
//! - Katz, "Mastering Audio," 3rd ed., 2015. Ch. 7 (mono compatibility,
//!   correlation metering).
//!
//! Part of #4264, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the stereo image optimizer.
///
/// Controls target width, correlation thresholds, bass mono summing,
/// and width smoothing. Use presets for common scenarios or build a
/// custom config with the builder methods.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StereoOptimizerConfig {
    /// Target stereo width in [0.0, 1.0]. 0.0 = mono, 1.0 = full width.
    /// Default: 0.7.
    pub target_width: f32,

    /// Minimum L/R correlation before the optimizer narrows the image.
    /// Range: [-1.0, 1.0]. Default: 0.3.
    pub min_correlation: f32,

    /// Frequencies below this threshold are forced mono (Hz).
    /// Set to 0.0 to disable bass mono summing. Default: 200.0.
    pub bass_mono_freq_hz: f32,

    /// Width adjustment smoothing time constant in milliseconds.
    /// Higher values prevent audible pumping. Default: 50.0.
    pub width_smoothing_ms: f32,

    /// Enable bass mono summing. Default: true.
    pub enable_bass_mono: bool,

    /// Dry/wet mix. 0.0 = bypass (dry), 1.0 = fully processed.
    /// Default: 1.0.
    pub mix: f32,
}

impl Default for StereoOptimizerConfig {
    fn default() -> Self {
        Self {
            target_width: 0.7,
            min_correlation: 0.3,
            bass_mono_freq_hz: 200.0,
            width_smoothing_ms: 50.0,
            enable_bass_mono: true,
            mix: 1.0,
        }
    }
}

impl StereoOptimizerConfig {
    /// Create a config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target stereo width.
    #[must_use]
    pub fn with_target_width(mut self, width: f32) -> Self {
        self.target_width = width;
        self
    }

    /// Set minimum correlation threshold.
    #[must_use]
    pub fn with_min_correlation(mut self, corr: f32) -> Self {
        self.min_correlation = corr;
        self
    }

    /// Set bass mono frequency threshold.
    #[must_use]
    pub fn with_bass_mono_freq_hz(mut self, freq: f32) -> Self {
        self.bass_mono_freq_hz = freq;
        self
    }

    /// Set width smoothing time constant.
    #[must_use]
    pub fn with_width_smoothing_ms(mut self, ms: f32) -> Self {
        self.width_smoothing_ms = ms;
        self
    }

    /// Enable or disable bass mono summing.
    #[must_use]
    pub fn with_bass_mono(mut self, enable: bool) -> Self {
        self.enable_bass_mono = enable;
        self
    }

    /// Set dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix;
        self
    }

    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range
    /// or non-finite.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.target_width.is_finite() || !(0.0..=1.0).contains(&self.target_width) {
            return Err(KokoroError::InvalidConfig {
                field: "target_width",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.target_width,
                ),
            });
        }
        if !self.min_correlation.is_finite() || !(-1.0..=1.0).contains(&self.min_correlation) {
            return Err(KokoroError::InvalidConfig {
                field: "min_correlation",
                reason: format!(
                    "must be finite and in [-1.0, 1.0], got {}",
                    self.min_correlation,
                ),
            });
        }
        if !self.bass_mono_freq_hz.is_finite() || self.bass_mono_freq_hz < 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "bass_mono_freq_hz",
                reason: format!("must be finite and >= 0.0, got {}", self.bass_mono_freq_hz),
            });
        }
        if !self.width_smoothing_ms.is_finite() || self.width_smoothing_ms < 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "width_smoothing_ms",
                reason: format!("must be finite and >= 0.0, got {}", self.width_smoothing_ms),
            });
        }
        if !self.mix.is_finite() || !(0.0..=1.0).contains(&self.mix) {
            return Err(KokoroError::InvalidConfig {
                field: "mix",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.mix),
            });
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Presets
    // -----------------------------------------------------------------------

    /// Natural-sounding preset: moderate width, gentle limiting.
    ///
    /// Suitable for most chorus mixes. Preserves spatial cues while
    /// maintaining safe mono compatibility.
    #[must_use]
    pub fn natural() -> Self {
        Self {
            target_width: 0.7,
            min_correlation: 0.3,
            bass_mono_freq_hz: 200.0,
            width_smoothing_ms: 50.0,
            enable_bass_mono: true,
            mix: 1.0,
        }
    }

    /// Wide preset: maximum stereo spread with relaxed correlation limits.
    ///
    /// Best for headphone listening where mono compatibility is less
    /// critical. Pushes voices further apart in the stereo field.
    #[must_use]
    pub fn wide() -> Self {
        Self {
            target_width: 1.0,
            min_correlation: 0.1,
            bass_mono_freq_hz: 120.0,
            width_smoothing_ms: 30.0,
            enable_bass_mono: true,
            mix: 1.0,
        }
    }

    /// Broadcast-safe preset: conservative width, strict correlation.
    ///
    /// Guarantees mono compatibility for broadcast, streaming, and
    /// phone speaker playback. Higher bass mono frequency for tight
    /// low end.
    #[must_use]
    pub fn broadcast_safe() -> Self {
        Self {
            target_width: 0.5,
            min_correlation: 0.5,
            bass_mono_freq_hz: 300.0,
            width_smoothing_ms: 80.0,
            enable_bass_mono: true,
            mix: 1.0,
        }
    }

    /// Headphone preset: wide image optimized for binaural playback.
    ///
    /// No bass mono summing since headphones do not suffer from
    /// acoustic bass cancellation. Relaxed correlation for maximum
    /// spatial impression.
    #[must_use]
    pub fn headphone() -> Self {
        Self {
            target_width: 0.9,
            min_correlation: 0.05,
            bass_mono_freq_hz: 0.0,
            width_smoothing_ms: 20.0,
            enable_bass_mono: false,
            mix: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// One-pole lowpass filter for bass extraction
// ---------------------------------------------------------------------------

/// Single-pole lowpass for splitting bass from the full signal.
#[derive(Debug, Clone)]
struct BassLpf {
    g: f32,
    z1_l: f32,
    z1_r: f32,
}

impl BassLpf {
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

    /// Process one stereo sample, returning the lowpass (bass) component.
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

    fn reset(&mut self) {
        self.z1_l = 0.0;
        self.z1_r = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Stereo optimizer processor
// ---------------------------------------------------------------------------

/// Real-time stereo image optimizer for mixed chorus output.
///
/// Monitors L/R correlation and dynamically adjusts mid/side balance
/// to maintain the configured width and correlation targets. Optionally
/// forces bass frequencies to mono.
///
/// # Usage
///
/// ```rust,no_run
/// use nn_models::kokoro_chorus_stereo_optimizer::*;
///
/// let config = StereoOptimizerConfig::natural();
/// let mut opt = StereoOptimizer::new(&config, 24000).unwrap();
///
/// let mut left  = vec![0.0f32; 1024];
/// let mut right = vec![0.0f32; 1024];
/// // ... fill with chorus stereo output ...
/// opt.process_stereo(&mut left, &mut right);
/// assert!(opt.current_correlation() >= -1.0);
/// ```
pub struct StereoOptimizer {
    config: StereoOptimizerConfig,
    sample_rate: f32,

    /// EMA of L/R correlation for smooth tracking.
    smoothed_correlation: f32,

    /// Current effective width (smoothed toward target or narrowed).
    effective_width: f32,

    /// Per-sample EMA coefficient for correlation tracking.
    /// Scaled to per-block alpha in `process_stereo` via
    /// `1 - (1 - a)^block_size`.
    corr_alpha: f32,

    /// Per-sample EMA coefficient for width adjustment.
    /// Scaled to per-block alpha in `process_stereo` via
    /// `1 - (1 - a)^block_size`.
    width_alpha: f32,

    /// Bass extraction filter for mono summing.
    bass_lpf: BassLpf,
}

impl StereoOptimizer {
    /// Create a new stereo optimizer.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails or
    /// sample rate is out of range.
    pub fn new(config: &StereoOptimizerConfig, sample_rate: u32) -> Result<Self, KokoroError> {
        config.validate()?;
        let sr = sample_rate as f32;
        if !(1000.0..=192000.0).contains(&sr) {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be in [1000, 192000], got {sample_rate}"),
            });
        }

        // Correlation tracking EMA: ~50ms time constant.
        let corr_window_samples = (sr * 0.05).max(1.0);
        let corr_alpha = (2.0 / (corr_window_samples + 1.0)).clamp(0.0001, 1.0);

        // Width smoothing EMA from config.
        let width_window_samples = (sr * config.width_smoothing_ms / 1000.0).max(1.0);
        let width_alpha = (2.0 / (width_window_samples + 1.0)).clamp(0.0001, 1.0);

        let bass_lpf = BassLpf::new(config.bass_mono_freq_hz, sr);

        Ok(Self {
            config: config.clone(),
            sample_rate: sr,
            smoothed_correlation: 1.0,
            effective_width: config.target_width,
            corr_alpha,
            width_alpha,
            bass_lpf,
        })
    }

    /// Process stereo audio in-place: analyze and optimize width/correlation.
    ///
    /// Applies mid/side width control, correlation-based narrowing, and
    /// optional bass mono summing. Both slices must have the same length.
    pub fn process_stereo(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        if len == 0 {
            return;
        }

        // Scale per-sample alpha to per-block: alpha_block = 1 - (1 - a)^N.
        // This preserves the configured time constant regardless of block size.
        let corr_alpha_block = per_block_alpha(self.corr_alpha, len);
        let width_alpha_block = per_block_alpha(self.width_alpha, len);

        // Measure block correlation to update the EMA.
        let block_corr = block_correlation(left, right, len);
        self.smoothed_correlation =
            ema_update(self.smoothed_correlation, block_corr, corr_alpha_block);

        // Determine target width: if correlation is too low, narrow.
        let desired_width = if self.smoothed_correlation < self.config.min_correlation {
            // Linearly reduce width as correlation drops below threshold.
            let headroom = self.config.min_correlation - self.smoothed_correlation;
            let reduction =
                (headroom / self.config.min_correlation.abs().max(0.01)).clamp(0.0, 1.0);
            self.config.target_width * (1.0 - reduction)
        } else {
            self.config.target_width
        };

        // Smooth width transitions to avoid pumping.
        self.effective_width = ema_update(self.effective_width, desired_width, width_alpha_block);

        let width = self.effective_width;
        let mix = self.config.mix;

        for i in 0..len {
            let l = left[i];
            let r = right[i];

            if !l.is_finite() || !r.is_finite() {
                left[i] = 0.0;
                right[i] = 0.0;
                continue;
            }

            // Decode to mid/side.
            let mid = (l + r) * 0.5;
            let side = (l - r) * 0.5;

            // Apply width: scale side by effective_width.
            let side_scaled = side * width;

            // Reconstruct L/R.
            let new_l = mid + side_scaled;
            let new_r = mid - side_scaled;

            // Apply bass mono summing if enabled.
            let (final_l, final_r) =
                if self.config.enable_bass_mono && self.config.bass_mono_freq_hz > 0.0 {
                    let (bass_l, bass_r) = self.bass_lpf.process(new_l, new_r);
                    let high_l = new_l - bass_l;
                    let high_r = new_r - bass_r;
                    let bass_mono = (bass_l + bass_r) * 0.5;
                    (bass_mono + high_l, bass_mono + high_r)
                } else {
                    (new_l, new_r)
                };

            // Dry/wet mix.
            left[i] = l + mix * (final_l - l);
            right[i] = r + mix * (final_r - r);

            // Clamp non-finite output.
            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
    }

    /// Get the current smoothed L/R correlation.
    #[must_use]
    pub fn current_correlation(&self) -> f32 {
        self.smoothed_correlation
    }

    /// Get the current effective stereo width after smoothing and limiting.
    #[must_use]
    pub fn effective_width(&self) -> f32 {
        self.effective_width
    }

    /// Get the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &StereoOptimizerConfig {
        &self.config
    }

    /// Reset all internal state (filters, smoothed values).
    ///
    /// Call between non-contiguous audio segments to avoid filter
    /// ringing and stale correlation estimates.
    pub fn reset(&mut self) {
        self.smoothed_correlation = 1.0;
        self.effective_width = self.config.target_width;
        self.bass_lpf.reset();
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute block-level Pearson correlation of L and R channels.
fn block_correlation(left: &[f32], right: &[f32], len: usize) -> f32 {
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
        let l = left[i];
        let r = right[i];
        if !l.is_finite() || !r.is_finite() {
            continue;
        }
        let ld = f64::from(l);
        let rd = f64::from(r);
        sum_l += ld;
        sum_r += rd;
        sum_ll += ld * ld;
        sum_rr += rd * rd;
        sum_lr += ld * rd;
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
        return 1.0;
    }

    (numerator / denom).clamp(-1.0, 1.0) as f32
}

/// Scale a per-sample EMA alpha to a per-block alpha for `block_size` samples.
///
/// Formula: `alpha_block = 1 - (1 - alpha_sample)^block_size`.
/// This ensures the configured time constant is respected regardless of
/// how many samples are processed per `process_stereo` call.
#[inline]
fn per_block_alpha(alpha_sample: f32, block_size: usize) -> f32 {
    if block_size == 0 {
        return 0.0;
    }
    let retention = (1.0 - alpha_sample).powi(block_size as i32);
    (1.0 - retention).clamp(0.0, 1.0)
}

/// Exponential moving average update: `state + alpha * (value - state)`.
#[inline]
fn ema_update(state: f32, value: f32, alpha: f32) -> f32 {
    let result = state + alpha * (value - state);
    if result.is_finite() {
        result
    } else {
        state
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_stereo_optimizer_tests.rs"]
mod tests;
