// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spectral balance analyzer and automatic mix engine for Kokoro chorus.
//!
//! Continuously monitors the overall frequency balance of the mixed chorus
//! output and adjusts per-voice gains to achieve a target spectral profile.
//! Think of it as an automatic mixing engineer: it listens to the mix in
//! 8 frequency bands, compares against a target balance curve, identifies
//! which voices contribute most to over- or under-represented bands, and
//! gently nudges per-voice gains to correct the balance.
//!
//! # Algorithm
//!
//! 1. Analyze mixed output in 8 frequency bands via windowed FFT.
//! 2. Compare measured band levels to the target balance profile.
//! 3. If `per_voice_analysis` is enabled, measure each voice's per-band
//!    contribution to identify which voices drive the imbalance.
//! 4. Compute per-voice gain adjustments to correct the deviation.
//! 5. Rate-limit adjustments: never move faster than `correction_speed`,
//!    never exceed `max_gain_change_db`.
//! 6. Lead voice protection: the designated lead voice is only boosted,
//!    never attenuated.
//!
//! # References
//!
//! - EBU R128 "Loudness normalisation and permitted maximum level of audio
//!   signals." European Broadcasting Union, 2020.
//! - Bristow-Johnson, R. "Audio EQ Cookbook." (2005).

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of analysis bands.
const NUM_BANDS: usize = 8;

/// Band edge frequencies (Hz) defining 8 bands for analysis.
/// Band 0: 0 -- 100 Hz (sub-bass)
/// Band 1: 100 -- 250 Hz (bass)
/// Band 2: 250 -- 600 Hz (low-mid)
/// Band 3: 600 -- 1200 Hz (mid)
/// Band 4: 1200 -- 2500 Hz (upper-mid)
/// Band 5: 2500 -- 5000 Hz (presence)
/// Band 6: 5000 -- 10000 Hz (brilliance)
/// Band 7: 10000+ Hz (air)
const BAND_EDGES: [f32; 9] = [
    0.0, 100.0, 250.0, 600.0, 1200.0, 2500.0, 5000.0, 10000.0, 24000.0,
];

/// Minimum representable level in dB.
const SILENCE_DB: f32 = -96.0;

/// Floor for linear amplitude to avoid log10(0).
const AMPLITUDE_FLOOR: f32 = 1e-12;

// ---------------------------------------------------------------------------
// Target balance
// ---------------------------------------------------------------------------

/// Target spectral balance profile for the auto-mixer.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TargetBalance {
    /// Flat response: all bands at 0 dB relative.
    Flat,
    /// Speech-optimized: mid-presence boost for clarity.
    Speech,
    /// Singing-optimized: scooped mids, high presence, low warmth.
    Singing,
    /// Broadcast: EBU R128 target curve (flat with gentle HF rolloff).
    Broadcast,
    /// Custom profile: (band_index, relative_db) pairs for 8 bands.
    Custom(Vec<(f32, f32)>),
}

impl TargetBalance {
    /// Evaluate the target balance as 8 relative dB values.
    fn evaluate(&self) -> [f32; NUM_BANDS] {
        match self {
            Self::Flat => [0.0; NUM_BANDS],
            Self::Speech => [
                -2.0, // sub-bass: reduce
                0.0,  // bass: neutral
                1.0,  // low-mid: slight warmth
                2.0,  // mid: presence boost
                2.5,  // upper-mid: clarity
                1.5,  // presence: articulation
                0.0,  // brilliance: neutral
                -2.0, // air: gentle rolloff
            ],
            Self::Singing => [
                1.0,  // sub-bass: warmth
                1.5,  // bass: body
                0.0,  // low-mid: neutral
                -1.5, // mid: scoop
                -1.0, // upper-mid: light scoop
                2.0,  // presence: projection
                1.5,  // brilliance: shimmer
                0.5,  // air: open top
            ],
            Self::Broadcast => [
                -1.0, // sub-bass: controlled
                0.0,  // bass: neutral
                0.0,  // low-mid: neutral
                0.5,  // mid: slight lift
                0.5,  // upper-mid: slight lift
                0.0,  // presence: neutral
                -1.0, // brilliance: rolloff
                -3.0, // air: rolloff
            ],
            Self::Custom(pairs) => {
                let mut levels = [0.0f32; NUM_BANDS];
                for &(band_f, db) in pairs {
                    let idx = band_f as usize;
                    if idx < NUM_BANDS && db.is_finite() {
                        levels[idx] = db;
                    }
                }
                levels
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the spectral balance auto-mixer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AutoMixConfig {
    /// Target spectral balance profile.
    pub target_balance: TargetBalance,
    /// Analysis window duration in milliseconds. Default: 100.
    pub analysis_window_ms: f32,
    /// Correction speed (0.0 = no correction, 1.0 = instant). Default: 0.3.
    pub correction_speed: f32,
    /// Maximum per-voice gain change in dB. Default: 6.0.
    pub max_gain_change_db: f32,
    /// Whether to analyze per-voice spectral contributions. Default: true.
    pub per_voice_analysis: bool,
    /// Index of the lead voice (protected from attenuation). Default: None.
    pub lead_voice: Option<usize>,
    /// Sample rate in Hz. Default: 24000.0.
    pub sample_rate: f32,
}

impl Default for AutoMixConfig {
    fn default() -> Self {
        Self {
            target_balance: TargetBalance::Flat,
            analysis_window_ms: 100.0,
            correction_speed: 0.3,
            max_gain_change_db: 6.0,
            per_voice_analysis: true,
            lead_voice: None,
            sample_rate: 24000.0,
        }
    }
}

impl AutoMixConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target spectral balance.
    #[must_use]
    pub fn with_target_balance(mut self, v: TargetBalance) -> Self {
        self.target_balance = v;
        self
    }

    /// Set the analysis window duration in milliseconds.
    #[must_use]
    pub fn with_analysis_window_ms(mut self, v: f32) -> Self {
        self.analysis_window_ms = v;
        self
    }

    /// Set the correction speed (0.0-1.0).
    #[must_use]
    pub fn with_correction_speed(mut self, v: f32) -> Self {
        self.correction_speed = v;
        self
    }

    /// Set the maximum per-voice gain change in dB.
    #[must_use]
    pub fn with_max_gain_change_db(mut self, v: f32) -> Self {
        self.max_gain_change_db = v;
        self
    }

    /// Set whether per-voice analysis is enabled.
    #[must_use]
    pub fn with_per_voice_analysis(mut self, v: bool) -> Self {
        self.per_voice_analysis = v;
        self
    }

    /// Set the lead voice index (protected from attenuation).
    #[must_use]
    pub fn with_lead_voice(mut self, v: Option<usize>) -> Self {
        self.lead_voice = v;
        self
    }

    /// Set the sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, v: f32) -> Self {
        self.sample_rate = v;
        self
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let err =
            |field: &'static str, reason: String| Err(KokoroError::InvalidConfig { field, reason });
        if !self.analysis_window_ms.is_finite()
            || self.analysis_window_ms < 10.0
            || self.analysis_window_ms > 1000.0
        {
            return err(
                "analysis_window_ms",
                format!(
                    "analysis_window_ms = {}: must be finite in [10.0, 1000.0]",
                    self.analysis_window_ms,
                ),
            );
        }
        if !self.correction_speed.is_finite() || !(0.0..=1.0).contains(&self.correction_speed) {
            return err(
                "correction_speed",
                format!(
                    "correction_speed = {}: must be finite in [0.0, 1.0]",
                    self.correction_speed,
                ),
            );
        }
        if !self.max_gain_change_db.is_finite()
            || self.max_gain_change_db < 0.0
            || self.max_gain_change_db > 24.0
        {
            return err(
                "max_gain_change_db",
                format!(
                    "max_gain_change_db = {}: must be finite in [0.0, 24.0]",
                    self.max_gain_change_db,
                ),
            );
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return err(
                "sample_rate",
                format!(
                    "sample_rate = {}: must be finite and positive",
                    self.sample_rate,
                ),
            );
        }
        Ok(())
    }

    // -- Presets -------------------------------------------------------------

    /// Balanced preset: moderate correction, no lead voice priority.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            target_balance: TargetBalance::Speech,
            correction_speed: 0.3,
            max_gain_change_db: 6.0,
            per_voice_analysis: true,
            lead_voice: None,
            ..Self::default()
        }
    }

    /// Lead-focused preset: protects voice 0 as lead, gentle correction.
    #[must_use]
    pub fn lead_focused() -> Self {
        Self {
            target_balance: TargetBalance::Speech,
            correction_speed: 0.2,
            max_gain_change_db: 4.0,
            per_voice_analysis: true,
            lead_voice: Some(0),
            ..Self::default()
        }
    }

    /// Blended preset: strong correction for tight ensemble cohesion.
    #[must_use]
    pub fn blended() -> Self {
        Self {
            target_balance: TargetBalance::Flat,
            correction_speed: 0.5,
            max_gain_change_db: 8.0,
            per_voice_analysis: true,
            lead_voice: None,
            ..Self::default()
        }
    }

    /// Broadcast preset: EBU R128 target with conservative correction.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            target_balance: TargetBalance::Broadcast,
            correction_speed: 0.25,
            max_gain_change_db: 4.0,
            per_voice_analysis: true,
            lead_voice: None,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Mix analysis result
// ---------------------------------------------------------------------------

/// Result of a spectral balance analysis and adjustment pass.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MixAnalysis {
    /// Measured level (dB) per frequency band in the mixed output.
    pub per_band_level: [f32; NUM_BANDS],
    /// Deviation from target balance (dB) per band. Positive = over target.
    pub target_deviation: [f32; NUM_BANDS],
    /// Gain adjustments (dB) applied to each voice.
    pub applied_gains: Vec<f32>,
    /// Lead voice clarity metric (0.0-1.0). Higher = lead sits above mix.
    pub lead_clarity: f32,
}

impl MixAnalysis {
    /// Create a new empty analysis for a given number of voices.
    fn new(n_voices: usize) -> Self {
        Self {
            per_band_level: [SILENCE_DB; NUM_BANDS],
            target_deviation: [0.0; NUM_BANDS],
            applied_gains: vec![0.0; n_voices],
            lead_clarity: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// FFT utilities (self-contained radix-2 DIT, matches chorus_auto_eq.rs)
// ---------------------------------------------------------------------------

fn fft(data: &mut [(f32, f32)]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            data.swap(i, j);
        }
    }
    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let angle_step = -std::f32::consts::TAU / stage_len as f32;
        for k in (0..n).step_by(stage_len) {
            for j in 0..half {
                let angle = angle_step * j as f32;
                let (tw_re, tw_im) = (angle.cos(), angle.sin());
                let (a_re, a_im) = data[k + j];
                let (b_re, b_im) = data[k + j + half];
                let t_re = b_re * tw_re - b_im * tw_im;
                let t_im = b_re * tw_im + b_im * tw_re;
                data[k + j] = (a_re + t_re, a_im + t_im);
                data[k + j + half] = (a_re - t_re, a_im - t_im);
            }
        }
        stage_len *= 2;
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            0.5 * (1.0 - (std::f32::consts::TAU * t).cos())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Band-level analysis
// ---------------------------------------------------------------------------

/// Measure per-band energy (dB) of an audio buffer using windowed FFT.
fn analyze_bands(audio: &[f32], window: &[f32], sample_rate: f32) -> [f32; NUM_BANDS] {
    let n = window.len();
    let mut frame = vec![0.0f32; n];
    let src_len = audio.len().min(n);
    let src_start = audio.len().saturating_sub(n);
    frame[..src_len].copy_from_slice(&audio[src_start..src_start + src_len]);

    for (s, &w) in frame.iter_mut().zip(window.iter()) {
        *s *= w;
    }

    let mut spectrum: Vec<(f32, f32)> = frame.iter().map(|&s| (s, 0.0)).collect();
    fft(&mut spectrum);

    let n_bins = n / 2 + 1;
    let bin_width = sample_rate / n as f32;

    let mut band_db = [SILENCE_DB; NUM_BANDS];
    for band in 0..NUM_BANDS {
        let lo = BAND_EDGES[band];
        let hi = BAND_EDGES[band + 1].min(sample_rate / 2.0);
        let bin_lo = ((lo / bin_width).ceil() as usize).max(1);
        let bin_hi = ((hi / bin_width).floor() as usize).min(n_bins - 1);
        if bin_lo > bin_hi {
            continue;
        }
        let mut energy = 0.0f32;
        let mut count = 0usize;
        for bin in bin_lo..=bin_hi {
            let (re, im) = spectrum[bin];
            energy += re * re + im * im;
            count += 1;
        }
        if count > 0 && energy > AMPLITUDE_FLOOR {
            let db = 20.0 * (energy / count as f32).sqrt().log10();
            band_db[band] = db.max(SILENCE_DB);
        }
    }
    band_db
}

// ---------------------------------------------------------------------------
// AutoMixer
// ---------------------------------------------------------------------------

/// Spectral balance auto-mixer for Kokoro chorus.
///
/// Monitors the mixed output in 8 frequency bands, compares against a
/// target spectral profile, and adjusts per-voice gains to correct the
/// balance. Adjustments are rate-limited and the lead voice is protected
/// from attenuation.
pub struct AutoMixer {
    config: AutoMixConfig,
    target_db: [f32; NUM_BANDS],
    window: Vec<f32>,
    /// Current per-voice gain offsets (dB). Smoothly converge toward targets.
    current_gains: Vec<f32>,
}

impl AutoMixer {
    /// Create a new auto-mixer.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range.
    pub fn new(config: &AutoMixConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let target_db = config.target_balance.evaluate();
        let window_samples =
            ((config.analysis_window_ms / 1000.0) * config.sample_rate).ceil() as usize;
        // Round up to next power of 2 for FFT.
        let fft_size = window_samples.next_power_of_two().clamp(256, 8192);
        let window = hann_window(fft_size);
        Ok(Self {
            config: config.clone(),
            target_db,
            window,
            current_gains: Vec::new(),
        })
    }

    /// Analyze spectral balance and adjust per-voice gains.
    ///
    /// `voices` is a mutable slice of per-voice audio buffers. The method:
    /// 1. Sums voices to form a mix, then analyzes band levels.
    /// 2. Computes deviation from target balance.
    /// 3. If per-voice analysis is enabled, identifies per-voice band
    ///    contributions to guide correction.
    /// 4. Applies smoothed gain adjustments to each voice.
    /// 5. Returns a [`MixAnalysis`] with measured levels and applied gains.
    pub fn analyze_and_adjust(&mut self, voices: &mut [Vec<f32>]) -> MixAnalysis {
        let n_voices = voices.len();
        if n_voices == 0 {
            return MixAnalysis::new(0);
        }

        // Ensure gain storage.
        self.ensure_capacity(n_voices);

        // Find common length.
        let max_len = voices.iter().map(Vec::len).max().unwrap_or(0);
        if max_len == 0 {
            return MixAnalysis::new(n_voices);
        }

        // Step 1: Sum voices into a mix buffer.
        let mut mix = vec![0.0f32; max_len];
        for voice in voices.iter() {
            for (i, &s) in voice.iter().enumerate() {
                if s.is_finite() {
                    mix[i] += s;
                }
            }
        }

        // Step 2: Analyze mixed output.
        let mix_bands = analyze_bands(&mix, &self.window, self.config.sample_rate);
        let mut analysis = MixAnalysis::new(n_voices);
        analysis.per_band_level = mix_bands;

        // Step 3: Compute deviation from target.
        for band in 0..NUM_BANDS {
            analysis.target_deviation[band] = mix_bands[band] - self.target_db[band];
        }

        // Step 4: Per-voice spectral contribution analysis.
        let voice_bands: Vec<[f32; NUM_BANDS]> = if self.config.per_voice_analysis {
            voices
                .iter()
                .map(|v| analyze_bands(v, &self.window, self.config.sample_rate))
                .collect()
        } else {
            vec![[SILENCE_DB; NUM_BANDS]; n_voices]
        };

        // Step 5: Compute per-voice gain adjustments.
        let speed = self.config.correction_speed;
        let max_db = self.config.max_gain_change_db;
        let lead = self.config.lead_voice;

        for vi in 0..n_voices {
            let mut desired_correction = 0.0f32;

            if self.config.per_voice_analysis {
                // For each band where the mix deviates from target, attribute
                // correction proportional to this voice's contribution.
                for band in 0..NUM_BANDS {
                    let deviation = analysis.target_deviation[band];
                    if deviation.abs() < 0.5 {
                        continue; // Within tolerance.
                    }
                    let voice_level = voice_bands[vi][band];
                    let mix_level = mix_bands[band];
                    // Voice contribution ratio: how much does this voice
                    // contribute to this band relative to the mix?
                    let contribution =
                        if mix_level > SILENCE_DB + 10.0 && voice_level > SILENCE_DB + 10.0 {
                            db_to_linear(voice_level) / db_to_linear(mix_level).max(AMPLITUDE_FLOOR)
                        } else {
                            1.0 / n_voices as f32
                        };
                    // Negative deviation = mix is below target = boost this voice.
                    // Positive deviation = mix is above target = cut this voice.
                    desired_correction -= deviation * contribution / NUM_BANDS as f32;
                }
            } else {
                // Without per-voice analysis, distribute correction equally.
                let total_deviation: f32 = analysis.target_deviation.iter().sum();
                desired_correction = -total_deviation / (n_voices as f32 * NUM_BANDS as f32);
            }

            // Clamp desired correction.
            desired_correction = desired_correction.clamp(-max_db, max_db);

            // Lead voice protection: never attenuate the lead.
            if lead == Some(vi) && desired_correction < 0.0 {
                desired_correction = 0.0;
            }

            // Smooth toward desired correction using correction_speed.
            let current = self.current_gains[vi];
            let target = desired_correction;
            let new_gain = current + (target - current) * speed;
            let new_gain = new_gain.clamp(-max_db, max_db);

            self.current_gains[vi] = if new_gain.is_finite() { new_gain } else { 0.0 };
            analysis.applied_gains[vi] = self.current_gains[vi];
        }

        // Step 6: Apply gain adjustments to voice buffers.
        for (vi, voice) in voices.iter_mut().enumerate() {
            let gain_db = self.current_gains[vi];
            if gain_db.abs() < 0.01 {
                continue; // Skip negligible adjustments.
            }
            let gain_linear = db_to_linear(gain_db);
            for s in voice.iter_mut() {
                if s.is_finite() {
                    *s *= gain_linear;
                    if !s.is_finite() {
                        *s = 0.0;
                    }
                }
            }
        }

        // Step 7: Lead clarity metric.
        analysis.lead_clarity = if let Some(lead_idx) = lead {
            if lead_idx < n_voices {
                compute_lead_clarity(&voice_bands, lead_idx)
            } else {
                0.0
            }
        } else {
            0.0
        };

        analysis
    }

    /// Get the current per-voice gain offsets (dB).
    #[must_use]
    pub fn current_gains(&self) -> &[f32] {
        &self.current_gains
    }

    /// Get the target spectral balance (dB per band).
    #[must_use]
    pub fn target_balance(&self) -> &[f32; NUM_BANDS] {
        &self.target_db
    }

    /// Reset all gain adjustments to zero.
    pub fn reset(&mut self) {
        self.current_gains.fill(0.0);
    }

    /// Ensure internal storage has capacity for `n_voices`.
    fn ensure_capacity(&mut self, n_voices: usize) {
        if self.current_gains.len() < n_voices {
            self.current_gains.resize(n_voices, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert dB to linear amplitude. Non-finite or silence returns 0.
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

/// Compute lead clarity: ratio of lead voice energy to average of other
/// voices across all bands. Returns 0.0--1.0 (1.0 = lead fully dominant).
fn compute_lead_clarity(voice_bands: &[[f32; NUM_BANDS]], lead_idx: usize) -> f32 {
    if voice_bands.len() < 2 || lead_idx >= voice_bands.len() {
        return 0.0;
    }
    let lead = &voice_bands[lead_idx];
    let n_others = voice_bands.len() - 1;
    let mut lead_energy = 0.0f32;
    let mut other_energy = 0.0f32;

    for band in 0..NUM_BANDS {
        let lead_lin = db_to_linear(lead[band]);
        lead_energy += lead_lin * lead_lin;

        for (vi, vb) in voice_bands.iter().enumerate() {
            if vi == lead_idx {
                continue;
            }
            let v_lin = db_to_linear(vb[band]);
            other_energy += v_lin * v_lin;
        }
    }

    let avg_other = other_energy / n_others as f32;
    let total = lead_energy + avg_other;
    if total < AMPLITUDE_FLOOR {
        return 0.0;
    }
    (lead_energy / total).clamp(0.0, 1.0)
}

#[cfg(test)]
#[path = "kokoro_chorus_auto_mix_tests.rs"]
mod tests;
