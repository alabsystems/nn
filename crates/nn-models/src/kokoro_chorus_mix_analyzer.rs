// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mix bus analyzer and auto-correction for professional chorus output.
//!
//! Monitors stereo chorus output in real-time and applies corrective
//! processing to ensure professional mix quality. This is the "mastering
//! engineer in a box" — it detects problems (phase issues, frequency
//! buildup, dynamic range issues) and applies gentle corrections.
//!
//! # Auto-correction chain
//!
//! When problems are detected, corrections are applied in this order:
//! 1. **DC offset removal** — subtract running mean
//! 2. **Phase correlation fix** — reduce stereo width toward mono
//! 3. **Bass mono-sum** — sum L/R below crossover via one-pole filter
//! 4. **Crest factor control** — soft knee compression on peaks
//! 5. **True peak limiting** — 4x oversampled ISP limiter
//! 6. **Loudness targeting** — gentle gain toward LUFS target
//!
//! # References
//!
//! - ITU-R BS.1770-4, "Algorithms to measure audio programme loudness."
//! - EBU R128, "Loudness normalisation and permitted maximum level."
//! - AES-6id-2006, "AES information document for digital audio — Personal
//!   computer audio quality measurements."
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

const SILENCE_DB: f32 = -120.0;
const AMPLITUDE_FLOOR: f64 = 1e-20;
const LUFS_OFFSET: f64 = -0.691;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the mix bus analyzer and auto-corrector.
///
/// Controls analysis thresholds and correction behavior. Use the builder
/// methods or one of the preset constructors (`transparent`, `broadcast`,
/// `streaming`, `aggressive`) for common configurations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MixAnalyzerConfig {
    /// Master enable for the analyzer + corrector. Default: true.
    pub enabled: bool,
    /// Target crest factor in dB (peak-to-RMS). Default: 12.0.
    pub crest_factor_target: f32,
    /// Minimum L/R correlation before correction engages. Default: 0.3.
    pub correlation_min: f32,
    /// Whether to remove DC offset. Default: true.
    pub dc_offset_removal: bool,
    /// ISP true peak ceiling in dBFS. Default: -1.0.
    pub true_peak_ceiling_db: f32,
    /// Target integrated LUFS. Default: -14.0.
    pub lufs_target: f32,
    /// Target spectral flatness (0.0 = tonal, 1.0 = noise). Default: 0.4.
    pub spectral_flatness_target: f32,
    /// How aggressively to correct (0.0 = off, 1.0 = max). Default: 0.3.
    pub correction_speed: f32,
    /// Mono-sum frequencies below this in Hz. Default: 120.0.
    pub bass_mono_below_hz: f32,
    /// Analysis window size in milliseconds. Default: 50.0.
    pub analyze_window_ms: f32,
}

impl Default for MixAnalyzerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            crest_factor_target: 12.0,
            correlation_min: 0.3,
            dc_offset_removal: true,
            true_peak_ceiling_db: -1.0,
            lufs_target: -14.0,
            spectral_flatness_target: 0.4,
            correction_speed: 0.3,
            bass_mono_below_hz: 120.0,
            analyze_window_ms: 50.0,
        }
    }
}

impl MixAnalyzerConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Minimal correction, mostly analysis.
    #[must_use]
    pub fn transparent() -> Self {
        Self {
            enabled: true,
            crest_factor_target: 18.0,
            correlation_min: -0.3,
            dc_offset_removal: true,
            true_peak_ceiling_db: -0.3,
            lufs_target: -14.0,
            spectral_flatness_target: 0.3,
            correction_speed: 0.05,
            bass_mono_below_hz: 80.0,
            analyze_window_ms: 50.0,
        }
    }

    /// Moderate correction, LUFS -14 target (broadcast standard).
    #[must_use]
    pub fn broadcast() -> Self {
        Self::default()
    }

    /// Conservative correction for streaming platforms (LUFS -16).
    #[must_use]
    pub fn streaming() -> Self {
        Self {
            enabled: true,
            crest_factor_target: 14.0,
            correlation_min: 0.2,
            dc_offset_removal: true,
            true_peak_ceiling_db: -1.5,
            lufs_target: -16.0,
            spectral_flatness_target: 0.4,
            correction_speed: 0.2,
            bass_mono_below_hz: 100.0,
            analyze_window_ms: 50.0,
        }
    }

    /// Heavy correction, narrow dynamic range.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            crest_factor_target: 8.0,
            correlation_min: 0.5,
            dc_offset_removal: true,
            true_peak_ceiling_db: -1.0,
            lufs_target: -14.0,
            spectral_flatness_target: 0.5,
            correction_speed: 0.7,
            bass_mono_below_hz: 150.0,
            analyze_window_ms: 30.0,
        }
    }

    // -- Builder methods --

    /// Set master enable.
    #[must_use]
    pub fn with_enabled(mut self, v: bool) -> Self {
        self.enabled = v;
        self
    }
    /// Set target crest factor in dB.
    #[must_use]
    pub fn with_crest_factor_target(mut self, v: f32) -> Self {
        self.crest_factor_target = v;
        self
    }
    /// Set minimum L/R correlation.
    #[must_use]
    pub fn with_correlation_min(mut self, v: f32) -> Self {
        self.correlation_min = v;
        self
    }
    /// Set DC offset removal.
    #[must_use]
    pub fn with_dc_offset_removal(mut self, v: bool) -> Self {
        self.dc_offset_removal = v;
        self
    }
    /// Set true peak ceiling in dBFS.
    #[must_use]
    pub fn with_true_peak_ceiling_db(mut self, v: f32) -> Self {
        self.true_peak_ceiling_db = v;
        self
    }
    /// Set target LUFS.
    #[must_use]
    pub fn with_lufs_target(mut self, v: f32) -> Self {
        self.lufs_target = v;
        self
    }
    /// Set spectral flatness target.
    #[must_use]
    pub fn with_spectral_flatness_target(mut self, v: f32) -> Self {
        self.spectral_flatness_target = v;
        self
    }
    /// Set correction speed (0.0-1.0).
    #[must_use]
    pub fn with_correction_speed(mut self, v: f32) -> Self {
        self.correction_speed = v;
        self
    }
    /// Set bass mono crossover frequency in Hz.
    #[must_use]
    pub fn with_bass_mono_below_hz(mut self, v: f32) -> Self {
        self.bass_mono_below_hz = v;
        self
    }
    /// Set analysis window size in ms.
    #[must_use]
    pub fn with_analyze_window_ms(mut self, v: f32) -> Self {
        self.analyze_window_ms = v;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let check = |field: &'static str, v: f32, lo: f32, hi: f32| -> Result<(), KokoroError> {
            if !v.is_finite() || v < lo || v > hi {
                return Err(KokoroError::InvalidConfig {
                    field,
                    reason: format!("must be finite in [{lo}, {hi}], got {v}"),
                });
            }
            Ok(())
        };
        check("crest_factor_target", self.crest_factor_target, 3.0, 30.0)?;
        check("correlation_min", self.correlation_min, -1.0, 1.0)?;
        check(
            "true_peak_ceiling_db",
            self.true_peak_ceiling_db,
            -12.0,
            0.0,
        )?;
        check("lufs_target", self.lufs_target, -60.0, 0.0)?;
        check(
            "spectral_flatness_target",
            self.spectral_flatness_target,
            0.0,
            1.0,
        )?;
        check("correction_speed", self.correction_speed, 0.0, 1.0)?;
        check("bass_mono_below_hz", self.bass_mono_below_hz, 0.0, 500.0)?;
        check("analyze_window_ms", self.analyze_window_ms, 5.0, 500.0)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Analysis snapshot
// ---------------------------------------------------------------------------

/// Snapshot of the current mix state from a single analysis window.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MixAnalysis {
    /// L/R RMS levels in dBFS.
    pub rms_db: (f32, f32),
    /// L/R peak levels in dBFS.
    pub peak_db: (f32, f32),
    /// L/R true peak levels in dBFS (4x oversampled).
    pub true_peak_db: (f32, f32),
    /// Crest factor in dB (peak-to-RMS ratio).
    pub crest_factor_db: f32,
    /// L/R correlation coefficient (-1.0 to 1.0).
    pub lr_correlation: f32,
    /// L/R DC offset.
    pub dc_offset: (f32, f32),
    /// Spectral centroid in Hz (brightness measure).
    pub spectral_centroid: f32,
    /// Spectral flatness (0.0 = tonal, 1.0 = noise-like).
    pub spectral_flatness: f32,
    /// Momentary loudness in LUFS.
    pub lufs_momentary: f32,
}

impl Default for MixAnalysis {
    fn default() -> Self {
        Self {
            rms_db: (SILENCE_DB, SILENCE_DB),
            peak_db: (SILENCE_DB, SILENCE_DB),
            true_peak_db: (SILENCE_DB, SILENCE_DB),
            crest_factor_db: 0.0,
            lr_correlation: 1.0,
            dc_offset: (0.0, 0.0),
            spectral_centroid: 0.0,
            spectral_flatness: 0.0,
            lufs_momentary: SILENCE_DB,
        }
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Mix bus analyzer and auto-corrector.
///
/// Monitors stereo audio for quality issues and optionally applies a
/// correction chain (DC removal, phase fix, bass mono, compression,
/// true-peak limiting, loudness targeting).
pub struct MixAnalyzerProcessor {
    config: MixAnalyzerConfig,
    sample_rate: f32,
    /// Running DC offset estimate (L, R) via exponential moving average.
    dc_state: (f64, f64),
    /// One-pole lowpass coefficient for bass mono crossover.
    bass_lp_coeff: f32,
    /// Bass filter state (L, R).
    bass_z1: (f32, f32),
    /// Envelope follower for crest-factor compression.
    comp_envelope: f32,
    /// True-peak limiter envelope.
    tp_envelope: f32,
    /// Smoothed gain for loudness targeting.
    lufs_gain: f32,
    /// Last computed analysis.
    last_analysis: Option<MixAnalysis>,
}

impl MixAnalyzerProcessor {
    /// Create a new processor.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config or sample_rate is invalid.
    pub fn new(config: &MixAnalyzerConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be finite and positive, got {sample_rate}"),
            });
        }
        let bass_lp_coeff = if config.bass_mono_below_hz > 0.0 {
            (1.0 - (-2.0 * std::f32::consts::PI * config.bass_mono_below_hz / sample_rate).exp())
                .clamp(0.001, 0.999)
        } else {
            0.0
        };
        Ok(Self {
            config: config.clone(),
            sample_rate,
            dc_state: (0.0, 0.0),
            bass_lp_coeff,
            bass_z1: (0.0, 0.0),
            comp_envelope: 0.0,
            tp_envelope: 0.0,
            lufs_gain: 1.0,
            last_analysis: None,
        })
    }

    /// Analyze stereo audio without modifying it.
    pub fn analyze(&mut self, left: &[f32], right: &[f32]) -> MixAnalysis {
        if !self.config.enabled || left.is_empty() || right.is_empty() {
            let a = MixAnalysis::default();
            self.last_analysis = Some(a);
            return a;
        }
        let len = left.len().min(right.len());
        let a = compute_analysis(&left[..len], &right[..len], self.sample_rate);
        self.last_analysis = Some(a);
        a
    }

    /// Analyze and auto-correct stereo audio in-place.
    ///
    /// Runs the full correction chain on problems detected by analysis.
    /// Corrections are applied only when analysis detects a problem
    /// exceeding the configured thresholds.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.config.enabled || left.is_empty() || right.is_empty() {
            self.last_analysis = Some(MixAnalysis::default());
            return;
        }
        let len = left.len().min(right.len());

        // 1. Analyze first (on original signal).
        let analysis = compute_analysis(&left[..len], &right[..len], self.sample_rate);
        self.last_analysis = Some(analysis);

        let speed = self.config.correction_speed;

        // 2. DC offset removal.
        if self.config.dc_offset_removal {
            self.remove_dc(&mut left[..len], &mut right[..len]);
        }

        // 3. Phase correlation fix.
        if analysis.lr_correlation < self.config.correlation_min {
            let amount = speed * (1.0 - analysis.lr_correlation.max(0.0));
            apply_width_reduction(&mut left[..len], &mut right[..len], amount);
        }

        // 4. Bass mono-sum.
        if self.config.bass_mono_below_hz > 0.0 {
            self.apply_bass_mono(&mut left[..len], &mut right[..len]);
        }

        // 5. Crest factor control (soft-knee compression).
        if analysis.crest_factor_db > self.config.crest_factor_target {
            let ratio = 1.0 + speed * 3.0; // 1:1 up to 1:4
            self.apply_crest_compression(&mut left[..len], &mut right[..len], ratio);
        }

        // 6. True peak limiting.
        let ceiling_lin = db_to_linear(self.config.true_peak_ceiling_db);
        self.apply_true_peak_limit(&mut left[..len], &mut right[..len], ceiling_lin);

        // 7. Loudness targeting.
        if analysis.lufs_momentary > SILENCE_DB + 10.0 {
            let diff_db = self.config.lufs_target - analysis.lufs_momentary;
            let target_gain = db_to_linear(diff_db.clamp(-6.0, 6.0));
            // Smooth gain change.
            let alpha = 0.001 + speed * 0.01;
            self.lufs_gain += alpha * (target_gain - self.lufs_gain);
            if self.lufs_gain.is_finite() && self.lufs_gain > 0.0 {
                for i in 0..len {
                    left[i] *= self.lufs_gain;
                    right[i] *= self.lufs_gain;
                    if !left[i].is_finite() {
                        left[i] = 0.0;
                    }
                    if !right[i].is_finite() {
                        right[i] = 0.0;
                    }
                }
            }
        }
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.dc_state = (0.0, 0.0);
        self.bass_z1 = (0.0, 0.0);
        self.comp_envelope = 0.0;
        self.tp_envelope = 0.0;
        self.lufs_gain = 1.0;
        self.last_analysis = None;
    }

    /// Retrieve the last computed analysis.
    #[must_use]
    pub fn last_analysis(&self) -> Option<&MixAnalysis> {
        self.last_analysis.as_ref()
    }

    /// Get a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &MixAnalyzerConfig {
        &self.config
    }

    /// Get the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    // -- Internal correction stages --

    /// Remove DC offset via running exponential average.
    fn remove_dc(&mut self, left: &mut [f32], right: &mut [f32]) {
        // Time constant: ~100 ms at sample_rate.
        let alpha = 1.0 / (f64::from(self.sample_rate) * 0.1);
        for i in 0..left.len().min(right.len()) {
            if left[i].is_finite() {
                self.dc_state.0 += alpha * (f64::from(left[i]) - self.dc_state.0);
                left[i] -= self.dc_state.0 as f32;
            }
            if right[i].is_finite() {
                self.dc_state.1 += alpha * (f64::from(right[i]) - self.dc_state.1);
                right[i] -= self.dc_state.1 as f32;
            }
            // Flush denormals.
            if self.dc_state.0.abs() < 1e-20 {
                self.dc_state.0 = 0.0;
            }
            if self.dc_state.1.abs() < 1e-20 {
                self.dc_state.1 = 0.0;
            }
        }
    }

    /// Apply bass mono-sum via one-pole lowpass crossover.
    fn apply_bass_mono(&mut self, left: &mut [f32], right: &mut [f32]) {
        let g = self.bass_lp_coeff;
        for i in 0..left.len().min(right.len()) {
            let l = left[i];
            let r = right[i];
            if !l.is_finite() || !r.is_finite() {
                left[i] = 0.0;
                right[i] = 0.0;
                self.bass_z1 = (0.0, 0.0);
                continue;
            }
            // Lowpass: extract bass.
            self.bass_z1.0 += g * (l - self.bass_z1.0);
            self.bass_z1.1 += g * (r - self.bass_z1.1);
            if self.bass_z1.0.abs() < 1e-20 {
                self.bass_z1.0 = 0.0;
            }
            if self.bass_z1.1.abs() < 1e-20 {
                self.bass_z1.1 = 0.0;
            }
            // Highpass residual.
            let high_l = l - self.bass_z1.0;
            let high_r = r - self.bass_z1.1;
            // Mono bass.
            let bass_mono = (self.bass_z1.0 + self.bass_z1.1) * 0.5;
            left[i] = bass_mono + high_l;
            right[i] = bass_mono + high_r;
            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
    }

    /// Soft-knee compression to control crest factor.
    fn apply_crest_compression(&mut self, left: &mut [f32], right: &mut [f32], ratio: f32) {
        let sr = f64::from(self.sample_rate);
        let attack = (-1.0 / (0.005 * sr)).exp() as f32; // 5 ms attack
        let release = (-1.0 / (0.050 * sr)).exp() as f32; // 50 ms release
        let thresh_lin = db_to_linear(-6.0); // -6 dBFS threshold

        for i in 0..left.len().min(right.len()) {
            let peak = left[i].abs().max(right[i].abs());
            if !peak.is_finite() {
                left[i] = 0.0;
                right[i] = 0.0;
                self.comp_envelope = 0.0;
                continue;
            }
            let coeff = if peak > self.comp_envelope {
                attack
            } else {
                release
            };
            self.comp_envelope = coeff * self.comp_envelope + (1.0 - coeff) * peak;
            if self.comp_envelope < 1e-20 {
                self.comp_envelope = 0.0;
            }

            if self.comp_envelope > thresh_lin {
                let over_db = linear_to_db(self.comp_envelope) - linear_to_db(thresh_lin);
                let reduction_db = over_db * (1.0 - 1.0 / ratio);
                let gain = db_to_linear(-reduction_db);
                if gain.is_finite() && gain > 0.0 {
                    left[i] *= gain;
                    right[i] *= gain;
                }
            }
            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
    }

    /// True peak limiter using 4x oversampled peak detection.
    fn apply_true_peak_limit(&mut self, left: &mut [f32], right: &mut [f32], ceiling: f32) {
        let sr = f64::from(self.sample_rate);
        let attack = (-1.0 / (0.0001 * sr)).exp() as f32; // 0.1 ms
        let release = (-1.0 / (0.050 * sr)).exp() as f32; // 50 ms

        for i in 0..left.len().min(right.len()) {
            let l = left[i];
            let r = right[i];
            if !l.is_finite() || !r.is_finite() {
                left[i] = 0.0;
                right[i] = 0.0;
                self.tp_envelope = 0.0;
                continue;
            }

            // Detect inter-sample peak via 4x Hermite interpolation.
            let tp = true_peak_pair(left, right, i);
            let coeff = if tp > self.tp_envelope {
                attack
            } else {
                release
            };
            self.tp_envelope = coeff * self.tp_envelope + (1.0 - coeff) * tp;
            if self.tp_envelope < 1e-20 {
                self.tp_envelope = 0.0;
            }

            if self.tp_envelope > ceiling {
                let gain = ceiling / self.tp_envelope;
                left[i] *= gain;
                right[i] *= gain;
            }

            // Hard safety clamp.
            left[i] = left[i].clamp(-ceiling, ceiling);
            right[i] = right[i].clamp(-ceiling, ceiling);
        }
    }
}

// ---------------------------------------------------------------------------
// Analysis computation (pure function)
// ---------------------------------------------------------------------------

/// Compute a full analysis snapshot from stereo buffers.
fn compute_analysis(left: &[f32], right: &[f32], sample_rate: f32) -> MixAnalysis {
    let len = left.len().min(right.len());
    if len == 0 {
        return MixAnalysis::default();
    }

    let mut sum_l = 0.0_f64;
    let mut sum_r = 0.0_f64;
    let mut sum_sq_l = 0.0_f64;
    let mut sum_sq_r = 0.0_f64;
    let mut peak_l: f32 = 0.0;
    let mut peak_r: f32 = 0.0;
    let mut sum_lr = 0.0_f64;
    let mut sum_ll = 0.0_f64;
    let mut sum_rr = 0.0_f64;
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
        sum_sq_l += ld * ld;
        sum_sq_r += rd * rd;
        sum_lr += ld * rd;
        sum_ll += ld * ld;
        sum_rr += rd * rd;
        if l.abs() > peak_l {
            peak_l = l.abs();
        }
        if r.abs() > peak_r {
            peak_r = r.abs();
        }
        count += 1;
    }

    if count == 0 {
        return MixAnalysis::default();
    }

    let n = count as f64;

    // RMS.
    let rms_l = (sum_sq_l / n).sqrt();
    let rms_r = (sum_sq_r / n).sqrt();
    let rms_db_l = amp_to_db(rms_l as f32);
    let rms_db_r = amp_to_db(rms_r as f32);

    // Peak.
    let peak_db_l = amp_to_db(peak_l);
    let peak_db_r = amp_to_db(peak_r);

    // True peak (4x oversampled).
    let tp_l = true_peak_mono(left);
    let tp_r = true_peak_mono(right);

    // Crest factor: peak / RMS.
    let rms_max = rms_l.max(rms_r);
    let peak_max = f64::from(peak_l.max(peak_r));
    let crest_factor_db = if rms_max > AMPLITUDE_FLOOR {
        (20.0 * (peak_max / rms_max).log10()) as f32
    } else {
        0.0
    };

    // L/R correlation (Pearson).
    let num = n * sum_lr - sum_l * sum_r;
    let den_l = (n * sum_ll - sum_l * sum_l).max(0.0).sqrt();
    let den_r = (n * sum_rr - sum_r * sum_r).max(0.0).sqrt();
    let den = den_l * den_r;
    let lr_correlation = if den > 1e-20 {
        (num / den).clamp(-1.0, 1.0) as f32
    } else {
        1.0
    };

    // DC offset.
    let dc_l = (sum_l / n) as f32;
    let dc_r = (sum_r / n) as f32;

    // Spectral centroid and flatness (mono sum for simplicity).
    let (centroid, flatness) = spectral_features(left, right, len, sample_rate);

    // Momentary LUFS (K-weighted approximation: use raw RMS for simplicity,
    // apply -0.691 offset per BS.1770).
    let mono_energy = ((sum_sq_l + sum_sq_r) * 0.5) / n;
    let lufs = if mono_energy > AMPLITUDE_FLOOR {
        let v = LUFS_OFFSET + 10.0 * mono_energy.log10();
        if v.is_finite() {
            v as f32
        } else {
            SILENCE_DB
        }
    } else {
        SILENCE_DB
    };

    MixAnalysis {
        rms_db: (rms_db_l, rms_db_r),
        peak_db: (peak_db_l, peak_db_r),
        true_peak_db: (tp_l, tp_r),
        crest_factor_db,
        lr_correlation,
        dc_offset: (dc_l, dc_r),
        spectral_centroid: centroid,
        spectral_flatness: flatness,
        lufs_momentary: lufs,
    }
}

// ---------------------------------------------------------------------------
// True peak (4x Hermite interpolation)
// ---------------------------------------------------------------------------

/// Measure true peak of a mono buffer via 4x cubic Hermite oversampling.
fn true_peak_mono(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return SILENCE_DB;
    }
    let mut max_abs: f64 = 0.0;
    for i in 0..audio.len() {
        let s = audio[i];
        if !s.is_finite() {
            continue;
        }
        let a = f64::from(s).abs();
        if a > max_abs {
            max_abs = a;
        }
        if i + 1 < audio.len() && audio[i + 1].is_finite() {
            let y0 = if i > 0 && audio[i - 1].is_finite() {
                f64::from(audio[i - 1])
            } else {
                f64::from(s)
            };
            let (y1, y2) = (f64::from(s), f64::from(audio[i + 1]));
            let y3 = if i + 2 < audio.len() && audio[i + 2].is_finite() {
                f64::from(audio[i + 2])
            } else {
                y2
            };
            for k in 1..4u32 {
                let t = f64::from(k) / 4.0;
                let c0 = y1;
                let c1 = 0.5 * (y2 - y0);
                let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
                let v = ((c3 * t + c2) * t + c1) * t + c0;
                if v.abs() > max_abs {
                    max_abs = v.abs();
                }
            }
        }
    }
    if max_abs < AMPLITUDE_FLOOR {
        return SILENCE_DB;
    }
    let db = 20.0 * max_abs.log10();
    if db.is_finite() {
        db as f32
    } else {
        SILENCE_DB
    }
}

/// Detect the true peak of a stereo pair at sample index `i` (4x oversampled).
fn true_peak_pair(left: &[f32], right: &[f32], i: usize) -> f32 {
    let len = left.len().min(right.len());
    let mut max_abs: f32 = left[i].abs().max(right[i].abs());

    for ch in &[left, right] {
        if i + 1 < len && ch[i].is_finite() && ch[i + 1].is_finite() {
            let y0 = if i > 0 && ch[i - 1].is_finite() {
                f64::from(ch[i - 1])
            } else {
                f64::from(ch[i])
            };
            let (y1, y2) = (f64::from(ch[i]), f64::from(ch[i + 1]));
            let y3 = if i + 2 < len && ch[i + 2].is_finite() {
                f64::from(ch[i + 2])
            } else {
                y2
            };
            for k in 1..4u32 {
                let t = f64::from(k) / 4.0;
                let c0 = y1;
                let c1 = 0.5 * (y2 - y0);
                let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
                let v = ((c3 * t + c2) * t + c1) * t + c0;
                let va = v.abs() as f32;
                if va > max_abs {
                    max_abs = va;
                }
            }
        }
    }
    max_abs
}

// ---------------------------------------------------------------------------
// Spectral features (centroid + flatness)
// ---------------------------------------------------------------------------

/// Compute spectral centroid (Hz) and flatness (0-1) from stereo audio.
fn spectral_features(left: &[f32], right: &[f32], len: usize, sample_rate: f32) -> (f32, f32) {
    // Use up to 2048 samples for FFT, mono sum.
    let n = len.clamp(64, 2048).next_power_of_two();
    let half = n / 2 + 1;
    let bin_hz = f64::from(sample_rate) / n as f64;

    // Power spectrum via DFT with Hann window (mono sum).
    let mut pspec = vec![0.0f64; half];
    for k in 0..half {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let fk = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
        for i in 0..len.min(n) {
            let l = if left[i].is_finite() {
                f64::from(left[i])
            } else {
                0.0
            };
            let r = if i < right.len() && right[i].is_finite() {
                f64::from(right[i])
            } else {
                0.0
            };
            let mono = (l + r) * 0.5;
            let w = if n > 1 {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
            } else {
                1.0
            };
            let x = mono * w;
            let ang = fk * i as f64;
            re += x * ang.cos();
            im += x * ang.sin();
        }
        pspec[k] = (re * re + im * im) / n as f64;
    }

    // Spectral centroid: weighted mean frequency.
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (k, &p) in pspec.iter().enumerate() {
        let freq = k as f64 * bin_hz;
        num += freq * p;
        den += p;
    }
    let centroid = if den > AMPLITUDE_FLOOR {
        (num / den) as f32
    } else {
        0.0
    };

    // Spectral flatness: geometric_mean / arithmetic_mean.
    let n_bins = pspec.len() as f64;
    let arith_mean = den / n_bins;
    let log_sum: f64 = pspec.iter().map(|&p| (p.max(AMPLITUDE_FLOOR)).ln()).sum();
    let geo_mean = (log_sum / n_bins).exp();
    let flatness = if arith_mean > AMPLITUDE_FLOOR {
        (geo_mean / arith_mean).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    (centroid, flatness)
}

// ---------------------------------------------------------------------------
// Stereo width reduction
// ---------------------------------------------------------------------------

/// Reduce stereo width by pulling side content toward mid.
fn apply_width_reduction(left: &mut [f32], right: &mut [f32], amount: f32) {
    let amount = amount.clamp(0.0, 1.0);
    let side_scale = 1.0 - amount;
    for i in 0..left.len().min(right.len()) {
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

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Convert amplitude to dBFS.
fn amp_to_db(amp: f32) -> f32 {
    if !amp.is_finite() || amp <= 0.0 {
        return SILENCE_DB;
    }
    let db = 20.0 * amp.log10();
    if db.is_finite() {
        db
    } else {
        SILENCE_DB
    }
}

/// Convert dBFS to linear amplitude.
fn db_to_linear(db: f32) -> f32 {
    if !db.is_finite() {
        return 0.0;
    }
    10.0f32.powf(db / 20.0)
}

/// Convert linear amplitude to dB.
fn linear_to_db(amp: f32) -> f32 {
    amp_to_db(amp)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_valid() {
        MixAnalyzerConfig::default()
            .validate()
            .expect("default should be valid");
    }

    #[test]
    fn test_config_presets_valid() {
        MixAnalyzerConfig::transparent()
            .validate()
            .expect("transparent");
        MixAnalyzerConfig::broadcast()
            .validate()
            .expect("broadcast");
        MixAnalyzerConfig::streaming()
            .validate()
            .expect("streaming");
        MixAnalyzerConfig::aggressive()
            .validate()
            .expect("aggressive");
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = MixAnalyzerConfig::new()
            .with_enabled(true)
            .with_crest_factor_target(10.0)
            .with_correlation_min(0.5)
            .with_dc_offset_removal(false)
            .with_true_peak_ceiling_db(-2.0)
            .with_lufs_target(-16.0)
            .with_spectral_flatness_target(0.3)
            .with_correction_speed(0.5)
            .with_bass_mono_below_hz(100.0)
            .with_analyze_window_ms(40.0);
        cfg.validate()
            .expect("builder chain should produce valid config");
        assert_eq!(cfg.crest_factor_target, 10.0);
        assert!(!cfg.dc_offset_removal);
    }

    #[test]
    fn test_config_invalid_correction_speed() {
        let cfg = MixAnalyzerConfig::new().with_correction_speed(1.5);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_analyze_silence() {
        let mut proc = MixAnalyzerProcessor::new(&MixAnalyzerConfig::default(), 24000.0)
            .expect("create processor");
        let left = vec![0.0f32; 1024];
        let right = vec![0.0f32; 1024];
        let a = proc.analyze(&left, &right);
        assert!(a.rms_db.0 <= SILENCE_DB + 1.0);
        assert!(a.rms_db.1 <= SILENCE_DB + 1.0);
    }

    #[test]
    fn test_analyze_mono_sine() {
        let mut proc = MixAnalyzerProcessor::new(&MixAnalyzerConfig::default(), 24000.0)
            .expect("create processor");
        let n = 2400;
        let freq = 440.0;
        let sr = 24000.0;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
            .collect();
        let a = proc.analyze(&signal, &signal);
        // Mono signal: correlation should be ~1.0.
        assert!(a.lr_correlation > 0.99, "correlation: {}", a.lr_correlation);
        // RMS should be around -6 dBFS for 0.5 amplitude sine.
        assert!(
            a.rms_db.0 > -10.0 && a.rms_db.0 < -2.0,
            "rms: {}",
            a.rms_db.0
        );
    }

    #[test]
    fn test_analyze_anticorrelated() {
        let mut proc = MixAnalyzerProcessor::new(&MixAnalyzerConfig::default(), 24000.0)
            .expect("create processor");
        let n = 2400;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.5)
            .collect();
        let inverted: Vec<f32> = signal.iter().map(|&s| -s).collect();
        let a = proc.analyze(&signal, &inverted);
        assert!(a.lr_correlation < -0.9, "correlation: {}", a.lr_correlation);
    }

    #[test]
    fn test_process_removes_dc() {
        let cfg = MixAnalyzerConfig::new()
            .with_dc_offset_removal(true)
            .with_correction_speed(0.0); // Only DC removal active.
        let mut proc = MixAnalyzerProcessor::new(&cfg, 24000.0).expect("create processor");
        let mut left: Vec<f32> = vec![0.5; 4800]; // DC offset of 0.5.
        let mut right: Vec<f32> = vec![0.5; 4800];
        proc.process(&mut left, &mut right);
        // After processing, mean should be closer to zero.
        let mean: f32 = left.iter().sum::<f32>() / left.len() as f32;
        assert!(mean.abs() < 0.3, "DC mean after removal: {mean}");
    }

    #[test]
    fn test_process_true_peak_limiting() {
        let cfg = MixAnalyzerConfig::new()
            .with_true_peak_ceiling_db(-3.0)
            .with_dc_offset_removal(false)
            .with_correction_speed(0.0);
        let mut proc = MixAnalyzerProcessor::new(&cfg, 24000.0).expect("create processor");
        let mut left: Vec<f32> = vec![0.9; 2400];
        let mut right: Vec<f32> = vec![0.9; 2400];
        proc.process(&mut left, &mut right);
        let ceiling = db_to_linear(-3.0);
        for &s in &left {
            assert!(
                s.abs() <= ceiling + 0.001,
                "sample {s} exceeds ceiling {ceiling}"
            );
        }
    }

    #[test]
    fn test_last_analysis() {
        let mut proc = MixAnalyzerProcessor::new(&MixAnalyzerConfig::default(), 24000.0)
            .expect("create processor");
        assert!(proc.last_analysis().is_none());
        let left = vec![0.1f32; 1024];
        let right = vec![0.1f32; 1024];
        proc.analyze(&left, &right);
        assert!(proc.last_analysis().is_some());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut proc = MixAnalyzerProcessor::new(&MixAnalyzerConfig::default(), 24000.0)
            .expect("create processor");
        let left = vec![0.1f32; 1024];
        let right = vec![0.1f32; 1024];
        proc.analyze(&left, &right);
        proc.reset();
        assert!(proc.last_analysis().is_none());
    }

    #[test]
    fn test_processor_invalid_sample_rate() {
        let result = MixAnalyzerProcessor::new(&MixAnalyzerConfig::default(), -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_empty_buffers() {
        let mut proc = MixAnalyzerProcessor::new(&MixAnalyzerConfig::default(), 24000.0)
            .expect("create processor");
        let mut left: Vec<f32> = vec![];
        let mut right: Vec<f32> = vec![];
        proc.process(&mut left, &mut right); // Should not panic.
    }

    #[test]
    fn test_process_nan_handling() {
        let mut proc = MixAnalyzerProcessor::new(&MixAnalyzerConfig::default(), 24000.0)
            .expect("create processor");
        let mut left = vec![f32::NAN; 512];
        let mut right = vec![f32::NAN; 512];
        proc.process(&mut left, &mut right);
        for &s in &left {
            assert!(!s.is_nan(), "NaN should be cleaned to 0.0");
        }
    }

    #[test]
    fn test_true_peak_mono_sine() {
        let n = 2400;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin() * 0.8)
            .collect();
        let tp = true_peak_mono(&signal);
        // True peak of a 0.8 amplitude sine should be near -1.9 dBFS.
        assert!(tp > -3.0 && tp < 0.0, "true peak: {tp}");
    }

    #[test]
    fn test_spectral_features_silence() {
        let left = vec![0.0f32; 256];
        let right = vec![0.0f32; 256];
        let (centroid, flatness) = spectral_features(&left, &right, 256, 24000.0);
        assert_eq!(centroid, 0.0);
        assert_eq!(flatness, 0.0);
    }

    #[test]
    fn test_db_conversions() {
        let lin = db_to_linear(-6.0);
        assert!((lin - 0.5012).abs() < 0.01, "db_to_linear(-6): {lin}");
        let db = amp_to_db(0.5);
        assert!((db - (-6.02)).abs() < 0.1, "amp_to_db(0.5): {db}");
    }

    #[test]
    fn test_width_reduction_full_mono() {
        let mut left = vec![1.0f32; 10];
        let mut right = vec![-1.0f32; 10];
        apply_width_reduction(&mut left, &mut right, 1.0);
        // Full collapse to mono: both should be 0.0 (mid of 1 and -1).
        for (&l, &r) in left.iter().zip(right.iter()) {
            assert!((l - 0.0).abs() < 1e-6);
            assert!((r - 0.0).abs() < 1e-6);
        }
    }
}
