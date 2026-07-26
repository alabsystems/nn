// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced sibilance processor for Kokoro chorus: de-essing, enhancement,
//! and multi-voice sibilant alignment.
//!
//! Three modes: **DeEss** (dynamic peaking EQ, 5-8 kHz), **Enhance** (presence
//! + air-band shelf), and **Balanced** (de-ess harsh peaks, boost gentle
//! presence). Multi-voice alignment micro-staggers sibilant onsets to prevent
//! phase cancellation.
//!
//! Ref: Giannoulis et al. JAES 60(6) 2012; Zolzer, DAFX 2nd ed. ch.2/5.
//! Part of #4264, #3351.

use crate::kokoro_chorus_saturation::db_to_linear;
use crate::kokoro_error::KokoroError;

/// Processing mode for sibilance control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SibilanceMode {
    /// Classic de-esser: reduce sibilance peaks.
    DeEss,
    /// Presence boost: enhance sibilance and air band.
    Enhance,
    /// Combined: reduce harsh peaks, boost gentle presence.
    Balanced,
}

/// Configuration for the sibilance processor.
///
/// Constructed via [`SibilanceConfig::new`] + builder methods (required for
/// cross-crate use due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SibilanceConfig {
    pub mode: SibilanceMode,
    pub detection_freq_hz: f32,
    pub detection_bandwidth_oct: f32,
    pub threshold_db: f32,
    pub reduction_db: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub enhancement_db: f32,
    pub air_freq_hz: f32,
    pub air_boost_db: f32,
    pub align_sibilants: bool,
    pub stagger_ms: f32,
}

impl Default for SibilanceConfig {
    fn default() -> Self {
        Self {
            mode: SibilanceMode::Balanced,
            detection_freq_hz: 6500.0,
            detection_bandwidth_oct: 1.5,
            threshold_db: -20.0,
            reduction_db: 6.0,
            attack_ms: 0.5,
            release_ms: 20.0,
            enhancement_db: 3.0,
            air_freq_hz: 12000.0,
            air_boost_db: 2.0,
            align_sibilants: true,
            stagger_ms: 1.0,
        }
    }
}

macro_rules! builder {
    ($name:ident, $field:ident, $ty:ty) => {
        #[must_use]
        pub fn $name(mut self, v: $ty) -> Self {
            self.$field = v;
            self
        }
    };
}

impl SibilanceConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    builder!(with_mode, mode, SibilanceMode);
    builder!(with_detection_freq_hz, detection_freq_hz, f32);
    builder!(with_detection_bandwidth_oct, detection_bandwidth_oct, f32);
    builder!(with_threshold_db, threshold_db, f32);
    builder!(with_reduction_db, reduction_db, f32);
    builder!(with_attack_ms, attack_ms, f32);
    builder!(with_release_ms, release_ms, f32);
    builder!(with_enhancement_db, enhancement_db, f32);
    builder!(with_air_freq_hz, air_freq_hz, f32);
    builder!(with_air_boost_db, air_boost_db, f32);
    builder!(with_align_sibilants, align_sibilants, bool);
    builder!(with_stagger_ms, stagger_ms, f32);

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let chk = |n: &'static str, v: f32, lo: f32, hi: f32| -> Result<(), KokoroError> {
            if !v.is_finite() || v < lo || v > hi {
                return Err(KokoroError::InvalidConfig {
                    field: n,
                    reason: format!("{n} = {v}: must be finite and in [{lo}, {hi}]"),
                });
            }
            Ok(())
        };
        chk("detection_freq_hz", self.detection_freq_hz, 1000.0, 20000.0)?;
        chk(
            "detection_bandwidth_oct",
            self.detection_bandwidth_oct,
            0.1,
            4.0,
        )?;
        chk("threshold_db", self.threshold_db, -60.0, 0.0)?;
        chk("reduction_db", self.reduction_db, 0.0, 24.0)?;
        chk("attack_ms", self.attack_ms, 0.01, 50.0)?;
        chk("release_ms", self.release_ms, 1.0, 500.0)?;
        chk("enhancement_db", self.enhancement_db, 0.0, 12.0)?;
        chk("air_freq_hz", self.air_freq_hz, 4000.0, 20000.0)?;
        chk("air_boost_db", self.air_boost_db, 0.0, 12.0)?;
        chk("stagger_ms", self.stagger_ms, 0.0, 10.0)?;
        Ok(())
    }
}

// -- Presets ----------------------------------------------------------------

/// Gentle de-essing: subtle sibilance taming, preserves natural brightness.
#[must_use]
pub fn gentle_deess() -> SibilanceConfig {
    SibilanceConfig::new()
        .with_mode(SibilanceMode::DeEss)
        .with_reduction_db(3.0)
        .with_threshold_db(-18.0)
        .with_attack_ms(1.0)
        .with_release_ms(30.0)
}

/// Aggressive de-essing: strong sibilance reduction for harsh recordings.
#[must_use]
pub fn aggressive_deess() -> SibilanceConfig {
    SibilanceConfig::new()
        .with_mode(SibilanceMode::DeEss)
        .with_reduction_db(12.0)
        .with_threshold_db(-24.0)
        .with_attack_ms(0.3)
        .with_release_ms(15.0)
}

/// Presence enhancement: add air and sparkle without harsh sibilance.
#[must_use]
pub fn presence_enhance() -> SibilanceConfig {
    SibilanceConfig::new()
        .with_mode(SibilanceMode::Enhance)
        .with_enhancement_db(4.0)
        .with_air_boost_db(3.0)
        .with_air_freq_hz(13000.0)
}

/// Broadcast-balanced: tame peaks while preserving clarity. Good default.
#[must_use]
pub fn broadcast_balanced() -> SibilanceConfig {
    SibilanceConfig::new()
        .with_mode(SibilanceMode::Balanced)
        .with_reduction_db(6.0)
        .with_enhancement_db(2.0)
        .with_threshold_db(-20.0)
        .with_air_boost_db(1.5)
}

// -- Biquad filter ----------------------------------------------------------

/// Second-order IIR (biquad) filter, direct form I.
#[derive(Debug, Clone)]
pub(crate) struct BiquadFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadFilter {
    fn bandpass(freq_hz: f32, bw_oct: f32, sr: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sr;
        let alpha = w0.sin() * (2.0_f32.ln() / 2.0 * bw_oct * w0 / w0.sin()).sinh();
        let a0 = 1.0 + alpha;
        Self::norm(alpha, 0.0, -alpha, -2.0 * w0.cos(), 1.0 - alpha, a0)
    }

    fn peaking(freq_hz: f32, q: f32, gain_db: f32, sr: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sr;
        let al = w0.sin() / (2.0 * q);
        let a0 = 1.0 + al / a;
        Self::norm(
            1.0 + al * a,
            -2.0 * w0.cos(),
            1.0 - al * a,
            -2.0 * w0.cos(),
            1.0 - al / a,
            a0,
        )
    }

    fn high_shelf(freq_hz: f32, gain_db: f32, sr: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sr;
        let c = w0.cos();
        let al = w0.sin() / 2.0 * 2.0_f32.sqrt();
        let sq = a.sqrt();
        let b0 = a * ((a + 1.0) + (a - 1.0) * c + 2.0 * sq * al);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * c);
        let b2 = a * ((a + 1.0) + (a - 1.0) * c - 2.0 * sq * al);
        let a0 = (a + 1.0) - (a - 1.0) * c + 2.0 * sq * al;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * c);
        let a2 = (a + 1.0) - (a - 1.0) * c - 2.0 * sq * al;
        Self::norm(b0, b1, b2, a1, a2, a0)
    }

    fn norm(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32, a0: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.reset();
            return 0.0;
        }
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = if y.is_finite() { y } else { 0.0 };
        self.y1
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// -- Sibilance processor ----------------------------------------------------

/// Stateful sibilance processor with detection, dynamic EQ, and air boost.
#[derive(Debug, Clone)]
pub struct SibilanceProcessor {
    config: SibilanceConfig,
    detection_filter: BiquadFilter,
    process_filter: BiquadFilter,
    air_filter: BiquadFilter,
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
    threshold_linear: f32,
    reduction_linear: f32,
    enhancement_linear: f32,
}

impl SibilanceProcessor {
    /// Create a new sibilance processor.
    pub fn new(config: SibilanceConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }
        let detection_filter = BiquadFilter::bandpass(
            config.detection_freq_hz,
            config.detection_bandwidth_oct,
            sample_rate,
        );
        let process_filter = BiquadFilter::peaking(config.detection_freq_hz, 2.0, 0.0, sample_rate);
        let air_filter =
            BiquadFilter::high_shelf(config.air_freq_hz, config.air_boost_db, sample_rate);
        let attack_coeff = (-1.0 / (config.attack_ms * 0.001 * sample_rate)).exp();
        let release_coeff = (-1.0 / (config.release_ms * 0.001 * sample_rate)).exp();

        Ok(Self {
            config,
            detection_filter,
            process_filter,
            air_filter,
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            threshold_linear: db_to_linear(config.threshold_db),
            reduction_linear: db_to_linear(-config.reduction_db),
            enhancement_linear: db_to_linear(config.enhancement_db),
        })
    }

    /// Create a processor using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: SibilanceConfig) -> Result<Self, KokoroError> {
        Self::new(config, crate::kokoro_tts::KOKORO_SAMPLE_RATE as f32)
    }

    /// Reset all internal filter and envelope state.
    pub fn reset(&mut self) {
        self.detection_filter.reset();
        self.process_filter.reset();
        self.air_filter.reset();
        self.envelope = 0.0;
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &SibilanceConfig {
        &self.config
    }

    /// Process a single voice audio buffer in-place.
    pub fn process_voice(&mut self, audio: &mut [f32]) {
        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }
            let dry = *sample;
            // Detection: bandpass -> rectify -> envelope
            let det = self.detection_filter.process(dry).abs();
            let coeff = if det > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * det;
            // Mode-dependent processing
            let out = match self.config.mode {
                SibilanceMode::DeEss => self.apply_deess(dry),
                SibilanceMode::Enhance => self.apply_enhance(dry),
                SibilanceMode::Balanced => self.apply_balanced(dry),
            };
            *sample = if out.is_finite() { out } else { 0.0 };
        }
    }

    #[inline]
    fn apply_deess(&mut self, x: f32) -> f32 {
        if self.envelope > self.threshold_linear {
            let overshoot = self.envelope / self.threshold_linear.max(1e-10);
            let gain = (1.0 / overshoot).max(self.reduction_linear).min(1.0);
            let sib = self.process_filter.process(x);
            (x - sib) + sib * gain
        } else {
            let _ = self.process_filter.process(x);
            x
        }
    }

    #[inline]
    fn apply_enhance(&mut self, x: f32) -> f32 {
        let sib = self.process_filter.process(x);
        let air = self.air_filter.process(x);
        if self.envelope < self.threshold_linear {
            (x - sib) + sib * self.enhancement_linear + (air - x)
        } else {
            x
        }
    }

    #[inline]
    fn apply_balanced(&mut self, x: f32) -> f32 {
        let sib = self.process_filter.process(x);
        let air = self.air_filter.process(x);
        let non_sib = x - sib;
        let harsh = self.threshold_linear * 2.0; // +6 dB
        if self.envelope > harsh {
            let os = self.envelope / harsh.max(1e-10);
            let g = (1.0 / os).max(self.reduction_linear).min(1.0);
            non_sib + sib * g
        } else if self.envelope < self.threshold_linear {
            let boost = 1.0 + (self.enhancement_linear - 1.0) * 0.5;
            non_sib + sib * boost + (air - x) * 0.5
        } else {
            x
        }
    }
}

// -- Multi-voice sibilant alignment -----------------------------------------

/// Detect sibilant onset positions (sample indices where energy crosses
/// the threshold upward).
fn detect_sibilant_onsets(audio: &[f32], cfg: &SibilanceConfig, sr: f32) -> Vec<usize> {
    let mut filt = BiquadFilter::bandpass(cfg.detection_freq_hz, cfg.detection_bandwidth_oct, sr);
    let thr = db_to_linear(cfg.threshold_db);
    let att = (-1.0 / (cfg.attack_ms * 0.001 * sr)).exp();
    let rel = (-1.0 / (cfg.release_ms * 0.001 * sr)).exp();
    let mut env = 0.0_f32;
    let mut was_above = false;
    let mut onsets = Vec::new();
    for (i, &s) in audio.iter().enumerate() {
        let d = filt.process(if s.is_finite() { s } else { 0.0 }).abs();
        let c = if d > env { att } else { rel };
        env = c * env + (1.0 - c) * d;
        let above = env > thr;
        if above && !was_above {
            onsets.push(i);
        }
        was_above = above;
    }
    onsets
}

/// Apply sample delay to audio via rotate + zero-fill.
fn apply_delay(audio: &mut [f32], delay: usize) {
    if delay == 0 || audio.is_empty() {
        return;
    }
    let d = delay.min(audio.len());
    audio.rotate_right(d);
    for s in audio.iter_mut().take(d) {
        *s = 0.0;
    }
}

/// Align sibilant timing across chorus voices. Voice 0 (lead) keeps
/// original timing; subsequent voices get progressive stagger offsets.
pub fn align_sibilants(voices: &mut [Vec<f32>], config: &SibilanceConfig, sample_rate: f32) {
    if voices.len() <= 1 || !config.align_sibilants || config.stagger_ms <= 0.0 {
        return;
    }
    let stagger = (config.stagger_ms * 0.001 * sample_rate).round() as usize;
    if stagger == 0 {
        return;
    }
    for (idx, voice) in voices.iter_mut().enumerate().skip(1) {
        let onsets = detect_sibilant_onsets(voice, config, sample_rate);
        if onsets.is_empty() {
            continue;
        }
        apply_delay(voice, stagger * idx);
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_sibilance_tests.rs"]
mod tests;
