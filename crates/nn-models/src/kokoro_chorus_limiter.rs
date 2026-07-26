// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! True peak limiter for Kokoro chorus final output.
//!
//! After mixing multiple TTS voices through the chorus pipeline (detune, EQ,
//! saturation, reverb, dynamics), the final signal can exceed 0 dBFS on
//! intersample peaks even when sample peaks are under the ceiling. A true peak
//! limiter with lookahead catches these overshoots and applies transparent gain
//! reduction to prevent clipping while preserving dynamics and loudness.
//!
//! # Algorithm
//!
//! 1. **Oversampled peak detection** — upsample the signal by 2x or 4x using
//!    4-point Hermite interpolation (same method as `measure_true_peak` in
//!    `kokoro_chorus_loudness`), then find the maximum absolute value across
//!    the upsampled signal.
//! 2. **Lookahead delay** — buffer incoming samples by `lookahead_ms` so the
//!    limiter can see peaks before they arrive, enabling a smooth attack ramp
//!    rather than hard clipping.
//! 3. **Gain computation** — for each sample, compute the gain reduction needed
//!    to bring the oversampled peak below the ceiling. Smooth with
//!    attack/release ballistics (exponential one-pole).
//! 4. **Stereo linking** — when enabled, left and right channels share the
//!    maximum gain reduction so the stereo image is preserved.
//! 5. **Wet/dry mix** — blend the limited signal with the dry signal.
//!
//! # References
//!
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design — A
//!   Tutorial and Analysis." JAES, 60(6), 2012.
//! - ITU-R BS.1770-4 "Algorithms to measure audio programme loudness and
//!   true-peak audio level." ITU, 2015.
//! - Zolzer, U. "DAFX — Digital Audio Effects." 2nd ed., Wiley, 2011, ch. 4.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert decibels to linear amplitude.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert linear amplitude to decibels. Returns `SILENCE_DB` for zero/negative.
#[inline]
fn linear_to_db(lin: f32) -> f32 {
    if lin <= 0.0 {
        SILENCE_DB
    } else {
        20.0 * lin.log10()
    }
}

const SILENCE_DB: f32 = -120.0;

/// 4-point Hermite interpolation at fractional position `t` in [0, 1)
/// between samples y1 and y2, with neighbors y0 and y3.
#[inline]
fn hermite_interp(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * t + c2) * t + c1) * t + c0
}

/// Compute the oversampled true peak of a single sample neighborhood.
/// Returns the maximum absolute value across the oversampled points.
fn oversampled_peak(y0: f32, y1: f32, y2: f32, y3: f32, factor: u32) -> f32 {
    let mut peak = y1.abs();
    for k in 1..factor {
        let t = k as f32 / factor as f32;
        let v = hermite_interp(y0, y1, y2, y3, t).abs();
        if v > peak {
            peak = v;
        }
    }
    peak
}

// ---------------------------------------------------------------------------
// LimiterConfig
// ---------------------------------------------------------------------------

/// Configuration for the true peak limiter.
///
/// Use the builder methods or presets to construct, then call [`validate()`]
/// before passing to [`LimiterProcessor::new`].
///
/// [`validate()`]: LimiterConfig::validate
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LimiterConfig {
    /// Output ceiling in dBFS. Must be in [-12.0, 0.0]. Default: -1.0.
    pub ceiling_db: f32,

    /// Attack time in milliseconds. Must be in [0.01, 10.0]. Default: 0.5.
    pub attack_ms: f32,

    /// Release time in milliseconds. Must be in [1.0, 1000.0]. Default: 50.0.
    pub release_ms: f32,

    /// Lookahead time in milliseconds. Must be in [0.0, 10.0]. Default: 1.0.
    ///
    /// Adds latency equal to this value but allows the limiter to anticipate
    /// peaks for a smoother, more transparent result.
    pub lookahead_ms: f32,

    /// Oversampling factor for intersample peak detection. 1, 2, or 4.
    /// Default: 2.
    pub oversample_factor: u32,

    /// Whether left and right channels share gain reduction. Default: true.
    pub stereo_link: bool,

    /// Wet/dry mix. 0.0 = fully dry (bypass), 1.0 = fully limited. Default: 1.0.
    pub mix: f32,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_db: -1.0,
            attack_ms: 0.5,
            release_ms: 50.0,
            lookahead_ms: 1.0,
            oversample_factor: 2,
            stereo_link: true,
            mix: 1.0,
        }
    }
}

impl LimiterConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Transparent preset: gentle limiting that preserves dynamics.
    pub fn transparent() -> Self {
        Self {
            ceiling_db: -1.0,
            attack_ms: 1.0,
            release_ms: 100.0,
            lookahead_ms: 1.5,
            oversample_factor: 4,
            stereo_link: true,
            mix: 1.0,
        }
    }

    /// Broadcast preset: moderate limiting for loudness compliance.
    pub fn broadcast() -> Self {
        Self {
            ceiling_db: -1.0,
            attack_ms: 0.5,
            release_ms: 50.0,
            lookahead_ms: 1.0,
            oversample_factor: 2,
            stereo_link: true,
            mix: 1.0,
        }
    }

    /// Aggressive preset: heavy limiting for maximum loudness.
    pub fn aggressive() -> Self {
        Self {
            ceiling_db: -0.3,
            attack_ms: 0.1,
            release_ms: 20.0,
            lookahead_ms: 0.5,
            oversample_factor: 4,
            stereo_link: true,
            mix: 1.0,
        }
    }

    /// Gentle preset: very light limiting, mostly for safety.
    pub fn gentle() -> Self {
        Self {
            ceiling_db: -1.5,
            attack_ms: 1.0,
            release_ms: 150.0,
            lookahead_ms: 2.0,
            oversample_factor: 2,
            stereo_link: true,
            mix: 1.0,
        }
    }

    // -- Builder methods ---------------------------------------------------

    /// Set the output ceiling in dBFS.
    #[must_use]
    pub fn with_ceiling_db(mut self, db: f32) -> Self {
        self.ceiling_db = db;
        self
    }

    /// Set the attack time in milliseconds.
    #[must_use]
    pub fn with_attack_ms(mut self, ms: f32) -> Self {
        self.attack_ms = ms;
        self
    }

    /// Set the release time in milliseconds.
    #[must_use]
    pub fn with_release_ms(mut self, ms: f32) -> Self {
        self.release_ms = ms;
        self
    }

    /// Set the lookahead time in milliseconds.
    #[must_use]
    pub fn with_lookahead_ms(mut self, ms: f32) -> Self {
        self.lookahead_ms = ms;
        self
    }

    /// Set the oversampling factor (1, 2, or 4).
    #[must_use]
    pub fn with_oversample_factor(mut self, factor: u32) -> Self {
        self.oversample_factor = factor;
        self
    }

    /// Set whether stereo linking is enabled.
    #[must_use]
    pub fn with_stereo_link(mut self, link: bool) -> Self {
        self.stereo_link = link;
        self
    }

    /// Set the wet/dry mix (0.0 to 1.0).
    #[must_use]
    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.ceiling_db.is_finite() || self.ceiling_db < -12.0 || self.ceiling_db > 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "ceiling_db",
                reason: format!(
                    "ceiling_db = {}: must be finite and in [-12, 0]",
                    self.ceiling_db,
                ),
            });
        }
        if !self.attack_ms.is_finite() || self.attack_ms < 0.01 || self.attack_ms > 10.0 {
            return Err(KokoroError::InvalidConfig {
                field: "attack_ms",
                reason: format!(
                    "attack_ms = {}: must be finite and in [0.01, 10]",
                    self.attack_ms,
                ),
            });
        }
        if !self.release_ms.is_finite() || self.release_ms < 1.0 || self.release_ms > 1000.0 {
            return Err(KokoroError::InvalidConfig {
                field: "release_ms",
                reason: format!(
                    "release_ms = {}: must be finite and in [1, 1000]",
                    self.release_ms,
                ),
            });
        }
        if !self.lookahead_ms.is_finite() || self.lookahead_ms < 0.0 || self.lookahead_ms > 10.0 {
            return Err(KokoroError::InvalidConfig {
                field: "lookahead_ms",
                reason: format!(
                    "lookahead_ms = {}: must be finite and in [0, 10]",
                    self.lookahead_ms,
                ),
            });
        }
        if !matches!(self.oversample_factor, 1 | 2 | 4) {
            return Err(KokoroError::InvalidConfig {
                field: "oversample_factor",
                reason: format!(
                    "oversample_factor = {}: must be 1, 2, or 4",
                    self.oversample_factor,
                ),
            });
        }
        if !self.mix.is_finite() || self.mix < 0.0 || self.mix > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "mix",
                reason: format!("mix = {}: must be finite and in [0, 1]", self.mix),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LimiterProcessor
// ---------------------------------------------------------------------------

/// True peak limiter processor with lookahead and stereo linking.
///
/// Maintains internal state (delay buffers, gain envelope) across calls to
/// [`process_stereo`]. Call [`reset`] to clear state when starting a new
/// audio segment.
///
/// [`process_stereo`]: LimiterProcessor::process_stereo
/// [`reset`]: LimiterProcessor::reset
pub struct LimiterProcessor {
    ceiling_linear: f32,
    attack_coeff: f32,
    release_coeff: f32,
    oversample_factor: u32,
    stereo_link: bool,
    mix: f32,

    /// Lookahead delay in samples.
    lookahead_samples: usize,

    /// Ring buffer for left channel lookahead delay.
    delay_left: Vec<f32>,
    /// Ring buffer for right channel lookahead delay.
    delay_right: Vec<f32>,
    /// Current write position in the ring buffer.
    delay_pos: usize,

    /// Current gain reduction envelope for left/linked channel (linear, <= 1.0).
    envelope_left: f32,
    /// Current gain reduction envelope for right channel (used when stereo_link is false).
    envelope_right: f32,

    /// Most recent gain reduction in dB (for metering).
    gr_db: f32,
}

impl LimiterProcessor {
    /// Create a new limiter processor from a validated config and sample rate.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if validation fails or if
    /// `sample_rate` is not positive and finite.
    pub fn new(config: &LimiterConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;

        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be positive and finite"),
            });
        }

        let sr = f64::from(sample_rate);
        let attack_coeff = (-1.0f64 / (f64::from(config.attack_ms) * 0.001 * sr)).exp() as f32;
        let release_coeff = (-1.0f64 / (f64::from(config.release_ms) * 0.001 * sr)).exp() as f32;

        let lookahead_samples = (f64::from(config.lookahead_ms) * 0.001 * sr).round() as usize;

        let delay_len = if lookahead_samples > 0 {
            lookahead_samples
        } else {
            1
        };

        Ok(Self {
            ceiling_linear: db_to_linear(config.ceiling_db),
            attack_coeff,
            release_coeff,
            oversample_factor: config.oversample_factor,
            stereo_link: config.stereo_link,
            mix: config.mix,
            lookahead_samples,
            delay_left: vec![0.0; delay_len],
            delay_right: vec![0.0; delay_len],
            delay_pos: 0,
            envelope_left: 1.0,
            envelope_right: 1.0,
            gr_db: 0.0,
        })
    }

    /// Process a stereo pair of buffers in place.
    ///
    /// Both buffers must have the same length. The limiter applies gain
    /// reduction to keep the oversampled true peak below the ceiling.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidInput` if the buffers have different lengths.
    pub fn process_stereo(
        &mut self,
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), KokoroError> {
        if left.len() != right.len() {
            return Err(KokoroError::InvalidInput(format!(
                "limiter stereo buffers must match: left={}, right={}",
                left.len(),
                right.len(),
            )));
        }

        if left.is_empty() {
            return Ok(());
        }

        let ceiling = self.ceiling_linear;
        let atk = self.attack_coeff;
        let rel = self.release_coeff;
        let os = self.oversample_factor;
        let mix = self.mix;

        // Bypass fast path.
        if mix <= 0.0 {
            return Ok(());
        }

        let n = left.len();
        let la = self.lookahead_samples;

        // Compute per-sample oversampled true peak per channel.
        let mut peak_left_buf = vec![0.0f32; n];
        let mut peak_right_buf = vec![0.0f32; n];
        for i in 0..n {
            let l = sanitize(left[i]);
            let r = sanitize(right[i]);

            if os > 1 {
                let l0 = if i > 0 { sanitize(left[i - 1]) } else { 0.0 };
                let l2 = if i + 1 < n { sanitize(left[i + 1]) } else { l };
                let l3 = if i + 2 < n { sanitize(left[i + 2]) } else { l2 };

                let r0 = if i > 0 { sanitize(right[i - 1]) } else { 0.0 };
                let r2 = if i + 1 < n { sanitize(right[i + 1]) } else { r };
                let r3 = if i + 2 < n {
                    sanitize(right[i + 2])
                } else {
                    r2
                };

                peak_left_buf[i] = oversampled_peak(l0, l, l2, l3, os);
                peak_right_buf[i] = oversampled_peak(r0, r, r2, r3, os);
            } else {
                peak_left_buf[i] = l.abs();
                peak_right_buf[i] = r.abs();
            }
        }

        let linked = self.stereo_link;

        // Apply gain reduction with lookahead.
        for i in 0..n {
            // Determine the peak to react to (look ahead into the future).
            let (la_peak_l, la_peak_r) = if la > 0 {
                let end = (i + la).min(n);
                let mut ml = peak_left_buf[i];
                let mut mr = peak_right_buf[i];
                for j in (i + 1)..end {
                    if peak_left_buf[j] > ml {
                        ml = peak_left_buf[j];
                    }
                    if peak_right_buf[j] > mr {
                        mr = peak_right_buf[j];
                    }
                }
                (ml, mr)
            } else {
                (peak_left_buf[i], peak_right_buf[i])
            };

            // When stereo-linked, use the max of both channels.
            let (peak_for_left, peak_for_right) = if linked {
                let m = la_peak_l.max(la_peak_r);
                (m, m)
            } else {
                (la_peak_l, la_peak_r)
            };

            // Compute target gain per channel.
            let target_l = if peak_for_left > ceiling && peak_for_left > 0.0 {
                ceiling / peak_for_left
            } else {
                1.0
            };
            let target_r = if peak_for_right > ceiling && peak_for_right > 0.0 {
                ceiling / peak_for_right
            } else {
                1.0
            };

            // Smooth envelopes with attack/release ballistics.
            let coeff_l = if target_l < self.envelope_left {
                atk
            } else {
                rel
            };
            self.envelope_left = coeff_l * self.envelope_left + (1.0 - coeff_l) * target_l;
            let coeff_r = if target_r < self.envelope_right {
                atk
            } else {
                rel
            };
            self.envelope_right = coeff_r * self.envelope_right + (1.0 - coeff_r) * target_r;

            // Clamp envelopes to [0, 1].
            self.envelope_left = clamp_envelope(self.envelope_left);
            self.envelope_right = clamp_envelope(self.envelope_right);

            let gain_l = self.envelope_left;
            let gain_r = self.envelope_right;

            // Write current samples to delay buffer and read delayed samples.
            let dl = self.delay_left[self.delay_pos];
            let dr = self.delay_right[self.delay_pos];
            self.delay_left[self.delay_pos] = sanitize(left[i]);
            self.delay_right[self.delay_pos] = sanitize(right[i]);
            self.delay_pos = (self.delay_pos + 1) % self.delay_left.len();

            // Apply gain and mix (use delayed signal for lookahead).
            if la > 0 {
                left[i] = mix * (dl * gain_l) + (1.0 - mix) * dl;
                right[i] = mix * (dr * gain_r) + (1.0 - mix) * dr;
            } else {
                let dry_l = sanitize(left[i]);
                let dry_r = sanitize(right[i]);
                left[i] = mix * (dry_l * gain_l) + (1.0 - mix) * dry_l;
                right[i] = mix * (dry_r * gain_r) + (1.0 - mix) * dry_r;
            }

            // Clamp to ceiling as hard safety net.
            if left[i].abs() > ceiling {
                left[i] = left[i].signum() * ceiling;
            }
            if right[i].abs() > ceiling {
                right[i] = right[i].signum() * ceiling;
            }
        }

        // Update gain reduction meter (report the deeper of the two channels).
        let min_env = self.envelope_left.min(self.envelope_right);
        self.gr_db = linear_to_db(min_env);

        Ok(())
    }

    /// Process a mono buffer in place.
    ///
    /// Convenience method that processes a single channel through the limiter.
    pub fn process_mono(&mut self, buffer: &mut [f32]) -> Result<(), KokoroError> {
        let mut right = vec![0.0f32; buffer.len()];
        // Copy buffer to right so stereo link sees the same signal.
        right.copy_from_slice(buffer);
        self.process_stereo(buffer, &mut right)?;
        Ok(())
    }

    /// Current gain reduction in dB (negative means attenuation).
    ///
    /// Returns 0.0 when no limiting is occurring. Returns a negative value
    /// (e.g., -3.0) when the limiter is reducing gain by that many dB.
    pub fn gain_reduction_db(&self) -> f32 {
        let min_env = self.envelope_left.min(self.envelope_right);
        if min_env >= 1.0 {
            0.0
        } else {
            self.gr_db
        }
    }

    /// Reset the processor state (delay buffers, envelope).
    ///
    /// Call this when starting a new audio segment to avoid artifacts from
    /// the previous segment's state.
    pub fn reset(&mut self) {
        self.delay_left.fill(0.0);
        self.delay_right.fill(0.0);
        self.delay_pos = 0;
        self.envelope_left = 1.0;
        self.envelope_right = 1.0;
        self.gr_db = 0.0;
    }

    /// The latency introduced by the lookahead in samples.
    pub fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }
}

/// Sanitize a sample: replace NaN/Inf with 0.0.
#[inline]
fn sanitize(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Clamp an envelope value to [0, 1], forcing 0 for NaN/Inf.
#[inline]
fn clamp_envelope(v: f32) -> f32 {
    if !v.is_finite() || v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_limiter_tests.rs"]
mod tests;
