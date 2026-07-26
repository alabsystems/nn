// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Formant-preserving pitch shift and spectral alignment for ensemble voice blending.
//!
//! When multiple TTS voices are mixed in a chorus, naive pitch shifting destroys
//! vowel quality (formant structure). This module uses a PSOLA-inspired
//! (Pitch-Synchronous Overlap-Add) approach: resampling within each pitch period
//! to shift F0 while preserving the spectral envelope (formants).
//!
//! References:
//! - Moulines & Charpentier, "Pitch-Synchronous Waveform Processing," 1990.
//! - de Cheveigne & Kawahara, "YIN frequency estimator," JASA, 2002.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for ensemble voice blending.
///
/// At `blend_strength = 0.0` no blending occurs. At `1.0` full
/// formant-preserving pitch correction and harmonic alignment are applied.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EnsembleBlendConfig {
    /// Overall blending strength. Range: [0.0, 1.0]. Default: `0.5`.
    pub blend_strength: f32,
    /// Use formant-preserving (PSOLA) pitch shift. Default: `true`.
    pub formant_preservation: bool,
    /// Align harmonic structures between voices. Default: `true`.
    pub harmonic_alignment: bool,
    /// Min pitch period in samples (~max F0). Range: [10, 500]. Default: 30.
    pub min_period: usize,
    /// Max pitch period in samples (~min F0). Range: [10, 2000]. Default: 300.
    pub max_period: usize,
}

impl Default for EnsembleBlendConfig {
    fn default() -> Self {
        Self {
            blend_strength: 0.5,
            formant_preservation: true,
            harmonic_alignment: true,
            min_period: 30,  // ~800 Hz at 24 kHz
            max_period: 300, // ~80 Hz at 24 kHz
        }
    }
}

impl EnsembleBlendConfig {
    /// Create a config with the given blend strength and all features enabled.
    pub fn new(blend_strength: f32) -> Result<Self, KokoroError> {
        let config = Self {
            blend_strength,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Create a disabled config (passthrough).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            blend_strength: 0.0,
            formant_preservation: false,
            harmonic_alignment: false,
            ..Self::default()
        }
    }

    /// Set formant preservation on/off.
    #[must_use]
    pub fn with_formant_preservation(mut self, enable: bool) -> Self {
        self.formant_preservation = enable;
        self
    }

    /// Set harmonic alignment on/off.
    #[must_use]
    pub fn with_harmonic_alignment(mut self, enable: bool) -> Self {
        self.harmonic_alignment = enable;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.blend_strength.is_finite() || !(0.0..=1.0).contains(&self.blend_strength) {
            return Err(KokoroError::InvalidConfig {
                field: "blend_strength",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.blend_strength
                ),
            });
        }
        if self.min_period < 10 || self.min_period > 500 {
            return Err(KokoroError::InvalidConfig {
                field: "min_period",
                reason: format!("must be in [10, 500], got {}", self.min_period),
            });
        }
        if self.max_period < 10 || self.max_period > 2000 {
            return Err(KokoroError::InvalidConfig {
                field: "max_period",
                reason: format!("must be in [10, 2000], got {}", self.max_period),
            });
        }
        if self.min_period > self.max_period {
            return Err(KokoroError::InvalidConfig {
                field: "max_period",
                reason: format!(
                    "must be >= min_period ({}), got {}",
                    self.min_period, self.max_period
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Formant-preserving pitch shift (PSOLA-style)
// ---------------------------------------------------------------------------

/// Formant-preserving pitch shifter using PSOLA overlap-add.
///
/// Resamples within each detected pitch period, preserving the spectral
/// envelope while changing F0. Unlike naive resampling which shifts formants.
pub struct FormantShift {
    min_period: usize,
    max_period: usize,
}

impl FormantShift {
    /// Create a new formant shifter with the given pitch period bounds.
    pub fn new(min_period: usize, max_period: usize) -> Self {
        let min = min_period.max(10);
        Self {
            min_period: min,
            max_period: max_period.max(min),
        }
    }

    /// Estimate pitch period via normalized autocorrelation. Returns `None` if unvoiced.
    fn estimate_period(&self, audio: &[f32], center: usize) -> Option<usize> {
        let half_win = self.max_period;
        let start = center.saturating_sub(half_win);
        let end = (center + half_win).min(audio.len());
        if end <= start + self.min_period {
            return None;
        }
        let window = &audio[start..end];
        let win_len = window.len();
        let energy: f64 = window.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        if energy < 1e-10 {
            return None;
        }

        let mut best_lag = self.min_period;
        let mut best_corr: f64 = -1.0;
        for lag in self.min_period..=self.max_period.min(win_len / 2) {
            let mut corr: f64 = 0.0;
            let mut energy_lag: f64 = 0.0;
            let n = win_len - lag;
            for i in 0..n {
                corr += f64::from(window[i]) * f64::from(window[i + lag]);
                energy_lag += f64::from(window[i + lag]) * f64::from(window[i + lag]);
            }
            let norm = (energy * energy_lag).sqrt();
            if norm < 1e-12 {
                continue;
            }
            let normalized = corr / norm;
            if normalized > best_corr {
                best_corr = normalized;
                best_lag = lag;
            }
        }
        if best_corr > 0.3 {
            Some(best_lag)
        } else {
            None
        }
    }

    /// Apply formant-preserving pitch shift. `pitch_factor` > 1.0 raises pitch.
    /// Returns a new buffer of the same length as input.
    pub fn shift(&self, audio: &[f32], pitch_factor: f32) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        if !pitch_factor.is_finite() || (pitch_factor - 1.0).abs() < 1e-6 {
            return audio.to_vec();
        }

        let len = audio.len();
        let mut output = vec![0.0f32; len];
        let mut window_sum = vec![0.0f32; len];
        let mut pos: usize = 0;

        while pos < len {
            let period = self
                .estimate_period(audio, pos)
                .unwrap_or(self.max_period / 2);
            let target_period = ((period as f32 / pitch_factor).round() as usize)
                .max(self.min_period)
                .min(self.max_period);

            let grain_start = pos.saturating_sub(period / 2);
            let grain_end = (grain_start + period).min(len);
            if grain_end - grain_start < 4 {
                pos += target_period.max(1);
                continue;
            }

            let resampled = resample_grain(&audio[grain_start..grain_end], target_period);
            let out_start = pos.saturating_sub(target_period / 2);
            for (i, &s) in resampled.iter().enumerate() {
                let out_idx = out_start + i;
                if out_idx >= len {
                    break;
                }
                let t = i as f32 / resampled.len().max(1) as f32;
                let hann = 0.5 * (1.0 - (std::f32::consts::TAU * t).cos());
                output[out_idx] += s * hann;
                window_sum[out_idx] += hann;
            }
            pos += target_period.max(1);
        }

        for (i, sample) in output.iter_mut().enumerate() {
            if window_sum[i] > 1e-6 {
                *sample /= window_sum[i];
            }
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
        output
    }
}

/// Resample a grain to `target_len` via linear interpolation. Core of formant
/// preservation: changing period length shifts F0 without altering spectral shape.
fn resample_grain(grain: &[f32], target_len: usize) -> Vec<f32> {
    if grain.is_empty() || target_len == 0 {
        return vec![0.0; target_len];
    }
    if grain.len() == target_len {
        return grain.to_vec();
    }

    let ratio = grain.len() as f64 / target_len as f64;
    (0..target_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let src_idx = src_pos as usize;
            let frac = (src_pos - src_idx as f64) as f32;
            let s0 = grain[src_idx.min(grain.len() - 1)];
            let s1 = if src_idx + 1 < grain.len() {
                grain[src_idx + 1]
            } else {
                s0
            };
            let val = s0 + frac * (s1 - s0);
            if val.is_finite() {
                val
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Spectral alignment
// ---------------------------------------------------------------------------

/// Aligns harmonic structures between voices to reduce inter-voice beating.
///
/// Computes spectral centroid per voice (via zero-crossing rate proxy) and
/// blends each voice's tonal balance toward the ensemble mean.
pub struct SpectralAlignment {
    window_size: usize,
}

impl SpectralAlignment {
    /// Create a new spectral aligner with the given analysis window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(64),
        }
    }

    /// Estimate normalized spectral centroid in [0.0, 1.0] via zero-crossing rate.
    pub fn estimate_centroid(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.5;
        }
        let mut crossings: usize = 0;
        let mut total_energy: f64 = 0.0;
        for chunk in audio.chunks(self.window_size) {
            for pair in chunk.windows(2) {
                if pair[0].signum() != pair[1].signum() {
                    crossings += 1;
                }
                total_energy += f64::from(pair[0]) * f64::from(pair[0]);
            }
            total_energy += f64::from(*chunk.last().unwrap_or(&0.0)).powi(2);
        }
        if total_energy < 1e-12 {
            return 0.5;
        }
        (crossings as f32 / (audio.len() - 1) as f32).clamp(0.0, 1.0)
    }

    /// Align a voice toward `target_centroid`. `strength` in [0.0, 1.0].
    pub fn align_voice(
        &self,
        audio: &mut [f32],
        current_centroid: f32,
        target_centroid: f32,
        strength: f32,
    ) {
        if audio.is_empty() || strength < 1e-6 {
            return;
        }
        let diff = target_centroid - current_centroid;
        if diff.abs() < 1e-4 {
            return;
        }
        let alpha = (diff.abs() * strength).clamp(0.0, 0.3);
        if diff < 0.0 {
            apply_lowpass_blend(audio, alpha);
        } else {
            apply_highpass_blend(audio, alpha);
        }
    }
}

/// Single-pole low-pass blend: y[n] = (1-a)*x[n] + a*y[n-1].
fn apply_lowpass_blend(audio: &mut [f32], alpha: f32) {
    if audio.is_empty() || alpha < 1e-7 {
        return;
    }
    let coeff = alpha.clamp(0.0, 0.5);
    let mut prev = audio[0];
    for sample in audio.iter_mut() {
        let filtered = (1.0 - coeff) * *sample + coeff * prev;
        *sample = if filtered.is_finite() { filtered } else { 0.0 };
        prev = *sample;
    }
}

/// First-order high-pass emphasis blended with original.
fn apply_highpass_blend(audio: &mut [f32], alpha: f32) {
    if audio.is_empty() || alpha < 1e-7 {
        return;
    }
    let coeff = alpha.clamp(0.0, 0.5);
    let mut prev_in = audio[0];
    let mut prev_out = audio[0];
    for sample in audio.iter_mut() {
        let hp = (1.0 - coeff) * (prev_out + *sample - prev_in);
        prev_in = *sample;
        let blended = *sample + coeff * (hp - *sample);
        *sample = if blended.is_finite() { blended } else { 0.0 };
        prev_out = hp;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the RMS energy of an audio buffer.
fn rms_energy(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = audio
        .iter()
        .map(|&s| {
            let v = f64::from(s);
            v * v
        })
        .sum();
    let rms = (sum_sq / audio.len() as f64).sqrt() as f32;
    if rms.is_finite() {
        rms
    } else {
        0.0
    }
}

/// Blend multiple voice audio buffers for natural ensemble mixing.
///
/// Applies formant-preserving pitch correction and spectral alignment
/// based on the configuration. Voices are modified in-place.
pub fn blend_voices(
    voices: &mut [Vec<f32>],
    config: &EnsembleBlendConfig,
    sample_rate: u32,
) -> Result<(), KokoroError> {
    config.validate()?;
    if voices.is_empty() || voices.len() == 1 || config.blend_strength < 1e-6 {
        return Ok(());
    }
    let _ = sample_rate; // Reserved for future sample-rate-dependent tuning.
    let strength = config.blend_strength;

    // --- Formant-preserving pitch alignment ---
    if config.formant_preservation {
        apply_formant_alignment(voices, config, strength);
    }

    // --- Spectral (harmonic) alignment ---
    if config.harmonic_alignment {
        let aligner = SpectralAlignment::new(2048);
        let centroids: Vec<f32> = voices
            .iter()
            .map(|v| aligner.estimate_centroid(v))
            .collect();
        let mean = centroids.iter().sum::<f32>() / centroids.len() as f32;
        let mean = if mean.is_finite() { mean } else { 0.5 };
        for (i, voice) in voices.iter_mut().enumerate() {
            aligner.align_voice(voice, centroids[i], mean, strength);
        }
    }
    Ok(())
}

/// Formant-preserving pitch alignment: nudge each voice's pitch toward the
/// ensemble median period while preserving formant structure and RMS energy.
fn apply_formant_alignment(voices: &mut [Vec<f32>], config: &EnsembleBlendConfig, strength: f32) {
    let shifter = FormantShift::new(config.min_period, config.max_period);

    let periods: Vec<Option<usize>> = voices
        .iter()
        .map(|v| {
            if v.len() < config.max_period * 4 {
                return None;
            }
            let positions = [v.len() / 4, v.len() / 2, 3 * v.len() / 4];
            let mut est = Vec::new();
            for &pos in &positions {
                if let Some(p) = shifter.estimate_period(v, pos) {
                    est.push(p);
                }
            }
            if est.is_empty() {
                None
            } else {
                est.sort_unstable();
                Some(est[est.len() / 2])
            }
        })
        .collect();

    let mut valid: Vec<usize> = periods.iter().filter_map(|p| *p).collect();
    if valid.len() < 2 {
        return;
    }
    valid.sort_unstable();
    let median = valid[valid.len() / 2];

    for (i, voice) in voices.iter_mut().enumerate() {
        if let Some(vp) = periods[i] {
            if vp == 0 {
                continue;
            }
            let ratio = median as f32 / vp as f32;
            if (ratio - 1.0).abs() > 0.01 {
                let factor = 1.0 + strength * (ratio - 1.0);
                let orig_rms = rms_energy(voice);
                let shifted = shifter.shift(voice, factor);
                let shift_rms = rms_energy(&shifted);
                *voice = shifted;
                if shift_rms > 1e-8 && orig_rms > 1e-8 {
                    let gain = orig_rms / shift_rms;
                    if gain.is_finite() {
                        for s in voice.iter_mut() {
                            *s *= gain;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_blend_tests.rs"]
mod tests;
