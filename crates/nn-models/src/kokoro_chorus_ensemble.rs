// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Classic ensemble chorus effect with modulated delay lines for Kokoro TTS.
//!
//! A DSP effect that creates lush, thick ensemble character by modulating
//! multiple delay lines with low-frequency oscillators (LFOs). Hardware
//! references: Roland Juno-60, Boss CE-1, string synthesizer ensemble.
//!
//! Each "voice" is a modulated delay line. An LFO varies the delay time,
//! creating pitch modulation via the Doppler effect. Multiple voices with
//! phase-offset LFOs produce the characteristic shimmer and width.
//!
//! References:
//! - Dattorro, "Effect Design Part 2", J. Audio Eng. Soc., 45(10), 1997.
//! - Pirkle, "Designing Audio Effect Plugins in C++", 2nd ed., Ch. 17.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Biquad lowpass (direct form II transposed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BiquadFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    fn lowpass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let omega = std::f32::consts::TAU * (cutoff_hz / sample_rate);
        let (sin_w, cos_w) = (omega.sin(), omega.cos());
        let alpha = sin_w / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0_inv = 1.0 / (1.0 + alpha);
        Self {
            b0: ((1.0 - cos_w) / 2.0) * a0_inv,
            b1: (1.0 - cos_w) * a0_inv,
            b2: ((1.0 - cos_w) / 2.0) * a0_inv,
            a1: (-2.0 * cos_w) * a0_inv,
            a2: (1.0 - alpha) * a0_inv,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        if y.is_finite() {
            y
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Ensemble effect operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum EnsembleMode {
    /// Classic chorus: sine LFO modulated delay (Roland Juno style).
    #[default]
    Chorus,
    /// String ensemble: multiple slow LFOs per voice, richer modulation.
    StringEnsemble,
    /// Flanger: short delay with feedback (jet-plane sweep).
    Flanger,
    /// Vibrato: 100% wet, pure pitch modulation only.
    Vibrato,
}


/// Configuration for the ensemble chorus effect.
///
/// Use builder methods or preset constructors. `#[non_exhaustive]` allows
/// adding fields without breaking downstream.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EnsembleConfig {
    /// Number of modulated delay lines. Range: [1, 8]. Default: 3.
    pub n_voices: usize,
    /// LFO rate in Hz. Range: [0.01, 20.0]. Default: 0.5.
    pub rate_hz: f32,
    /// Modulation depth in ms. Range: [0.1, 50.0]. Default: 5.0.
    pub depth_ms: f32,
    /// Feedback (0 = chorus, >0 = flange). Range: [0.0, 0.7]. Default: 0.0.
    pub feedback: f32,
    /// Wet/dry mix. Range: [0.0, 1.0]. Default: 0.5.
    pub mix: f32,
    /// LFO phase spread between voices (degrees). Default: 120.
    pub stereo_spread: f32,
    /// Effect mode. Default: Chorus.
    pub mode: EnsembleMode,
    /// High-cut on wet signal (Hz). Range: [200, 20000]. Default: 8000.
    pub high_cut_hz: f32,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            n_voices: 3,
            rate_hz: 0.5,
            depth_ms: 5.0,
            feedback: 0.0,
            mix: 0.5,
            stereo_spread: 120.0,
            mode: EnsembleMode::Chorus,
            high_cut_hz: 8000.0,
        }
    }
}

impl EnsembleConfig {
    /// Create a default config with validation.
    pub fn new() -> Result<Self, KokoroError> {
        let c = Self::default();
        c.validate()?;
        Ok(c)
    }

    #[must_use]
    pub fn with_n_voices(mut self, n: usize) -> Self {
        self.n_voices = n;
        self
    }
    #[must_use]
    pub fn with_rate_hz(mut self, v: f32) -> Self {
        self.rate_hz = v;
        self
    }
    #[must_use]
    pub fn with_depth_ms(mut self, v: f32) -> Self {
        self.depth_ms = v;
        self
    }
    #[must_use]
    pub fn with_feedback(mut self, v: f32) -> Self {
        self.feedback = v;
        self
    }
    #[must_use]
    pub fn with_mix(mut self, v: f32) -> Self {
        self.mix = v;
        self
    }
    #[must_use]
    pub fn with_stereo_spread(mut self, v: f32) -> Self {
        self.stereo_spread = v;
        self
    }
    #[must_use]
    pub fn with_mode(mut self, v: EnsembleMode) -> Self {
        self.mode = v;
        self
    }
    #[must_use]
    pub fn with_high_cut_hz(mut self, v: f32) -> Self {
        self.high_cut_hz = v;
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.n_voices < 1 || self.n_voices > 8 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: format!("must be in [1, 8], got {}", self.n_voices),
            });
        }
        check_range("rate_hz", self.rate_hz, 0.01, 20.0)?;
        check_range("depth_ms", self.depth_ms, 0.1, 50.0)?;
        check_range("feedback", self.feedback, 0.0, 0.7)?;
        check_range("mix", self.mix, 0.0, 1.0)?;
        check_range("stereo_spread", self.stereo_spread, 0.0, 360.0)?;
        check_range("high_cut_hz", self.high_cut_hz, 200.0, 20000.0)?;
        Ok(())
    }

    // -- Presets ---------------------------------------------------------------

    /// Subtle chorus: gentle shimmer, barely noticeable widening.
    #[must_use]
    pub fn subtle_chorus() -> Self {
        Self {
            n_voices: 2,
            rate_hz: 0.3,
            depth_ms: 3.0,
            feedback: 0.0,
            mix: 0.3,
            stereo_spread: 180.0,
            mode: EnsembleMode::Chorus,
            high_cut_hz: 10000.0,
        }
    }

    /// Rich ensemble: lush multi-voice chorus for vocal bus.
    #[must_use]
    pub fn rich_ensemble() -> Self {
        Self {
            n_voices: 4,
            rate_hz: 0.6,
            depth_ms: 7.0,
            feedback: 0.0,
            mix: 0.5,
            stereo_spread: 90.0,
            mode: EnsembleMode::Chorus,
            high_cut_hz: 8000.0,
        }
    }

    /// Thick flanger: metallic sweep with feedback.
    #[must_use]
    pub fn thick_flange() -> Self {
        Self {
            n_voices: 2,
            rate_hz: 0.2,
            depth_ms: 2.0,
            feedback: 0.5,
            mix: 0.5,
            stereo_spread: 180.0,
            mode: EnsembleMode::Flanger,
            high_cut_hz: 12000.0,
        }
    }

    /// String machine: slow, deep modulation like a Solina or ARP Omni.
    #[must_use]
    pub fn string_machine() -> Self {
        Self {
            n_voices: 6,
            rate_hz: 0.3,
            depth_ms: 8.0,
            feedback: 0.0,
            mix: 0.6,
            stereo_spread: 60.0,
            mode: EnsembleMode::StringEnsemble,
            high_cut_hz: 6000.0,
        }
    }

    /// Roland Juno Chorus I: single-voice, moderate rate.
    #[must_use]
    pub fn juno_chorus_i() -> Self {
        Self {
            n_voices: 1,
            rate_hz: 0.513,
            depth_ms: 4.0,
            feedback: 0.0,
            mix: 0.5,
            stereo_spread: 0.0,
            mode: EnsembleMode::Chorus,
            high_cut_hz: 9000.0,
        }
    }

    /// Roland Juno Chorus II: dual-voice, faster rate, wider.
    #[must_use]
    pub fn juno_chorus_ii() -> Self {
        Self {
            n_voices: 2,
            rate_hz: 0.863,
            depth_ms: 4.5,
            feedback: 0.0,
            mix: 0.5,
            stereo_spread: 180.0,
            mode: EnsembleMode::Chorus,
            high_cut_hz: 9000.0,
        }
    }
}

fn check_range(field: &'static str, val: f32, lo: f32, hi: f32) -> Result<(), KokoroError> {
    if !val.is_finite() || val < lo || val > hi {
        Err(KokoroError::InvalidConfig {
            field,
            reason: format!("must be finite and in [{lo}, {hi}], got {val}"),
        })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Ensemble chorus processor with modulated delay lines.
///
/// Maintains per-voice circular delay buffers and LFO phase accumulators.
#[derive(Debug, Clone)]
pub struct EnsembleProcessor {
    config: EnsembleConfig,
    delay_lines: Vec<Vec<f32>>,
    write_positions: Vec<usize>,
    lfo_phases: Vec<f32>,
    lfo_increments: Vec<f32>,
    lfo_phases_2: Vec<f32>, // secondary LFO for StringEnsemble
    lfo_increments_2: Vec<f32>,
    feedback_buffer: Vec<f32>,
    high_cut_filter_l: BiquadFilter,
    high_cut_filter_r: BiquadFilter,
    sample_rate: f32,
}

impl EnsembleProcessor {
    /// Create a new ensemble processor.
    pub fn new(config: &EnsembleConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate < 1000.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be >= 1000, got {sample_rate}"),
            });
        }
        let n = config.n_voices;
        let buf_len = (((config.depth_ms + 20.0) / 1000.0) * sample_rate) as usize + 4;
        let spread_rad = config.stereo_spread.to_radians();

        let mut lfo_phases = Vec::with_capacity(n);
        let mut lfo_inc = Vec::with_capacity(n);
        let mut lfo_phases_2 = Vec::with_capacity(n);
        let mut lfo_inc_2 = Vec::with_capacity(n);
        for i in 0..n {
            let phase = i as f32 * spread_rad;
            lfo_phases.push(phase % std::f32::consts::TAU);
            lfo_inc.push(config.rate_hz / sample_rate);
            lfo_phases_2.push((phase * 0.7) % std::f32::consts::TAU);
            lfo_inc_2.push((config.rate_hz * 1.12 + 0.02 * i as f32) / sample_rate);
        }
        Ok(Self {
            config: config.clone(),
            delay_lines: vec![vec![0.0_f32; buf_len]; n],
            write_positions: vec![0; n],
            lfo_phases,
            lfo_increments: lfo_inc,
            lfo_phases_2,
            lfo_increments_2: lfo_inc_2,
            feedback_buffer: vec![0.0; n],
            high_cut_filter_l: BiquadFilter::lowpass(config.high_cut_hz, sample_rate),
            high_cut_filter_r: BiquadFilter::lowpass(config.high_cut_hz, sample_rate),
            sample_rate,
        })
    }

    /// Reset all internal state (delay lines, LFOs, filters).
    pub fn reset(&mut self) {
        for dl in &mut self.delay_lines {
            dl.fill(0.0);
        }
        for wp in &mut self.write_positions {
            *wp = 0;
        }
        let spread_rad = self.config.stereo_spread.to_radians();
        for (i, p) in self.lfo_phases.iter_mut().enumerate() {
            *p = (i as f32 * spread_rad) % std::f32::consts::TAU;
        }
        for (i, p) in self.lfo_phases_2.iter_mut().enumerate() {
            *p = (i as f32 * spread_rad * 0.7) % std::f32::consts::TAU;
        }
        self.feedback_buffer.fill(0.0);
        self.high_cut_filter_l.reset();
        self.high_cut_filter_r.reset();
    }

    #[inline]
    fn lfo_value(&self, voice: usize) -> f32 {
        let phase = self.lfo_phases[voice];
        match self.config.mode {
            EnsembleMode::StringEnsemble => {
                phase.sin() * 0.6 + self.lfo_phases_2[voice].sin() * 0.4
            }
            _ => phase.sin(),
        }
    }

    #[inline]
    fn advance_lfo(&mut self) {
        let tau = std::f32::consts::TAU;
        for i in 0..self.config.n_voices {
            self.lfo_phases[i] = (self.lfo_phases[i] + self.lfo_increments[i] * tau) % tau;
            self.lfo_phases_2[i] = (self.lfo_phases_2[i] + self.lfo_increments_2[i] * tau) % tau;
        }
    }

    /// Read from delay line with cubic Hermite interpolation.
    #[inline]
    fn read_delay_cubic(&self, voice: usize, delay_samples: f32) -> f32 {
        let dl = &self.delay_lines[voice];
        let len = dl.len();
        let wp = self.write_positions[voice] as isize;
        let d = delay_samples.max(0.0);
        let d_int = d as isize;
        let frac = d - d_int as f32;

        let idx = |off: isize| -> f32 {
            dl[((wp - d_int - off) % len as isize + len as isize) as usize % len]
        };
        let (ym1, y0, y1, y2) = (idx(2), idx(1), idx(0), idx(-1));

        let c0 = y0;
        let c1 = 0.5 * (y1 - ym1);
        let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
        let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
        let r = ((c3 * frac + c2) * frac + c1) * frac + c0;
        if r.is_finite() {
            r
        } else {
            0.0
        }
    }

    #[inline]
    fn write_delay(&mut self, voice: usize, sample: f32) {
        let len = self.delay_lines[voice].len();
        let wp = self.write_positions[voice];
        self.delay_lines[voice][wp] = if sample.is_finite() { sample } else { 0.0 };
        self.write_positions[voice] = (wp + 1) % len;
    }

    fn base_delay_samples(&self) -> f32 {
        let ms = if self.config.mode == EnsembleMode::Flanger {
            2.0
        } else {
            10.0
        };
        ms * self.sample_rate / 1000.0
    }

    fn voice_pan(&self, voice: usize) -> f32 {
        if self.config.n_voices <= 1 {
            return 0.5;
        }
        voice as f32 / (self.config.n_voices - 1) as f32
    }

    /// Process one sample through all delay voices, returning (wet_l, wet_r).
    fn process_sample(&mut self, input: f32) -> (f32, f32) {
        let base = self.base_delay_samples();
        let depth_samp = self.config.depth_ms * self.sample_rate / 1000.0;
        let n = self.config.n_voices;
        let (mut wl, mut wr) = (0.0_f32, 0.0_f32);

        for v in 0..n {
            let delay = base + depth_samp * self.lfo_value(v);
            let delayed = self.read_delay_cubic(v, delay);
            self.write_delay(v, input + self.config.feedback * self.feedback_buffer[v]);
            self.feedback_buffer[v] = delayed;

            let angle = self.voice_pan(v) * std::f32::consts::FRAC_PI_2;
            wl += delayed * angle.cos();
            wr += delayed * angle.sin();
        }
        let norm = 1.0 / (n as f32).sqrt();
        self.advance_lfo();
        (wl * norm, wr * norm)
    }

    fn effective_mix(&self) -> f32 {
        if self.config.mode == EnsembleMode::Vibrato {
            1.0
        } else {
            self.config.mix
        }
    }

    /// Process a stereo audio bus in-place. Buffers must have equal length.
    pub fn process_stereo(
        &mut self,
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), KokoroError> {
        if left.len() != right.len() {
            return Err(KokoroError::InvalidInput(format!(
                "stereo buffer length mismatch: left={}, right={}",
                left.len(),
                right.len(),
            )));
        }
        let mix = self.effective_mix();
        let dry = 1.0 - mix;
        for i in 0..left.len() {
            let (wl, wr) = self.process_sample((left[i] + right[i]) * 0.5);
            left[i] = dry * left[i] + mix * self.high_cut_filter_l.process(wl);
            right[i] = dry * right[i] + mix * self.high_cut_filter_r.process(wr);
            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
        Ok(())
    }

    /// Process mono audio to stereo. Returns `(left, right)`.
    #[must_use]
    pub fn process_mono(&mut self, audio: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mix = self.effective_mix();
        let dry = 1.0 - mix;
        let mut out_l = Vec::with_capacity(audio.len());
        let mut out_r = Vec::with_capacity(audio.len());
        for &s in audio {
            let (wl, wr) = self.process_sample(s);
            let l = dry * s + mix * self.high_cut_filter_l.process(wl);
            let r = dry * s + mix * self.high_cut_filter_r.process(wr);
            out_l.push(if l.is_finite() { l } else { 0.0 });
            out_r.push(if r.is_finite() { r } else { 0.0 });
        }
        (out_l, out_r)
    }
}

// ---------------------------------------------------------------------------
// Tests (extracted to stay under 500-line limit)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_ensemble_tests.rs"]
mod tests;
