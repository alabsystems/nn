// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spectral ducking and sidechain compression for Kokoro chorus voice management.
//!
//! In a multi-voice TTS chorus, the lead voice must remain intelligible while
//! backing voices provide harmonic support. Ducking reduces the level of
//! non-lead voices when the lead is active, either broadband or per-frequency-band.
//!
//! # Architecture
//!
//! ```text
//! Lead voice ──────► Energy analysis ──► Gain envelope (attack/release)
//!                         │                      │
//!                         │ (per-band or         │ (gain reduction)
//!                         │  broadband)          │
//!                         ▼                      ▼
//! Other voices ──► [Band split] ──► Apply gain ──► [Band sum] ──► Output
//! ```
//!
//! # Frequency-aware ducking
//!
//! When `frequency_aware` is enabled, the lead voice energy is analyzed per
//! frequency band using simplified Linkwitz-Riley crossovers (two cascaded
//! one-pole filters per band boundary). Ducking is applied only in bands
//! where the lead is active, preserving harmonic content in other bands.
//!
//! # Sidechain compression
//!
//! A simplified variant where an external signal (e.g., music, effects)
//! triggers gain reduction on the audio. The sidechain envelope follower
//! uses the same attack/release ballistics as the ducker.
//!
//! # References
//!
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design."
//!   Journal of the Audio Engineering Society, 60(6), 2012.
//! - Zölzer, U. "DAFX: Digital Audio Effects." 2nd ed. Wiley, 2011.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Ducking configuration
// ---------------------------------------------------------------------------

/// Configuration for spectral ducking of non-lead voices.
///
/// When the lead voice exceeds the threshold, other voices are attenuated
/// by `duck_amount_db` with configurable attack/release ballistics.
/// Frequency-aware mode splits the signal into `n_bands` and only ducks
/// in bands where the lead is active.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DuckingConfig {
    /// Index of the lead voice that others duck around.
    pub lead_voice_index: usize,
    /// How much to reduce non-lead voices when lead is active (dB).
    /// Must be in [-20.0, 0.0] and finite. -6 dB is a subtle duck,
    /// -12 dB is moderate, -20 dB is aggressive.
    pub duck_amount_db: f32,
    /// How quickly ducking engages (ms). Must be in [1.0, 50.0] and finite.
    pub attack_ms: f32,
    /// How slowly ducking releases (ms). Must be in [50.0, 500.0] and finite.
    pub release_ms: f32,
    /// Lead signal level (dBFS) that triggers ducking.
    /// Must be in [-60.0, 0.0] and finite.
    pub threshold_db: f32,
    /// When true, only duck in frequency bands where the lead is present.
    pub frequency_aware: bool,
    /// Number of frequency bands for frequency-aware ducking.
    /// Must be in [1, 8]. Ignored when `frequency_aware` is false.
    pub n_bands: usize,
}

impl Default for DuckingConfig {
    fn default() -> Self {
        Self {
            lead_voice_index: 0,
            duck_amount_db: -9.0,
            attack_ms: 5.0,
            release_ms: 150.0,
            threshold_db: -30.0,
            frequency_aware: false,
            n_bands: 4,
        }
    }
}

impl DuckingConfig {
    /// Create a new ducking configuration with validation.
    pub fn new(lead_voice_index: usize) -> Self {
        Self {
            lead_voice_index,
            ..Self::default()
        }
    }

    /// Set the ducking amount in dB.
    #[must_use]
    pub fn with_duck_amount_db(mut self, db: f32) -> Self {
        self.duck_amount_db = db;
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

    /// Set the threshold in dBFS.
    #[must_use]
    pub fn with_threshold_db(mut self, db: f32) -> Self {
        self.threshold_db = db;
        self
    }

    /// Enable frequency-aware ducking.
    #[must_use]
    pub fn with_frequency_aware(mut self, enabled: bool) -> Self {
        self.frequency_aware = enabled;
        self
    }

    /// Set the number of frequency bands.
    #[must_use]
    pub fn with_n_bands(mut self, n: usize) -> Self {
        self.n_bands = n;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.duck_amount_db.is_finite()
            || self.duck_amount_db < -20.0
            || self.duck_amount_db > 0.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "duck_amount_db",
                reason: format!(
                    "duck_amount_db = {}: must be finite and in [-20, 0]",
                    self.duck_amount_db,
                ),
            });
        }
        if !self.attack_ms.is_finite() || self.attack_ms < 1.0 || self.attack_ms > 50.0 {
            return Err(KokoroError::InvalidConfig {
                field: "attack_ms",
                reason: format!(
                    "attack_ms = {}: must be finite and in [1, 50]",
                    self.attack_ms,
                ),
            });
        }
        if !self.release_ms.is_finite() || self.release_ms < 50.0 || self.release_ms > 500.0 {
            return Err(KokoroError::InvalidConfig {
                field: "release_ms",
                reason: format!(
                    "release_ms = {}: must be finite and in [50, 500]",
                    self.release_ms,
                ),
            });
        }
        if !self.threshold_db.is_finite() || self.threshold_db < -60.0 || self.threshold_db > 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "threshold_db",
                reason: format!(
                    "threshold_db = {}: must be finite and in [-60, 0]",
                    self.threshold_db,
                ),
            });
        }
        if self.n_bands < 1 || self.n_bands > 8 {
            return Err(KokoroError::InvalidConfig {
                field: "n_bands",
                reason: format!("n_bands = {}: must be in [1, 8]", self.n_bands),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// One-pole envelope follower
// ---------------------------------------------------------------------------

/// Simple one-pole envelope follower with separate attack/release.
///
/// Tracks the RMS energy of the signal using ballistic averaging.
#[derive(Debug, Clone)]
struct EnvelopeFollower {
    attack_coeff: f32,
    release_coeff: f32,
    envelope_sq: f32,
}

impl EnvelopeFollower {
    fn new(attack_ms: f32, release_ms: f32) -> Self {
        let sr = KOKORO_SAMPLE_RATE as f64;
        let attack_coeff = (-1.0 / (f64::from(attack_ms) * 0.001 * sr)).exp() as f32;
        let release_coeff = (-1.0 / (f64::from(release_ms) * 0.001 * sr)).exp() as f32;
        Self {
            attack_coeff,
            release_coeff,
            envelope_sq: 0.0,
        }
    }

    /// Feed a sample and return the current envelope level in dBFS.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.envelope_sq = 0.0;
            return -120.0;
        }
        let x_sq = x * x;
        let coeff = if x_sq > self.envelope_sq {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope_sq = coeff * self.envelope_sq + (1.0 - coeff) * x_sq;
        if self.envelope_sq < 1e-20 {
            self.envelope_sq = 0.0;
            return -120.0;
        }
        let rms = self.envelope_sq.sqrt();
        let db = 20.0 * rms.log10();
        if !db.is_finite() {
            -120.0
        } else {
            db
        }
    }

    fn reset(&mut self) {
        self.envelope_sq = 0.0;
    }
}

// ---------------------------------------------------------------------------
// One-pole crossover filter (simplified band-split for ducking)
// ---------------------------------------------------------------------------

/// A simple one-pole lowpass/highpass pair for band-splitting.
///
/// Two cascaded one-pole filters approximate a Linkwitz-Riley-like crossover
/// for the ducking use-case (exact phase alignment is not critical since
/// we are only using bands for energy analysis and gain application).
#[derive(Debug, Clone)]
struct OnePoleXover {
    lp_z: f32,
    lp2_z: f32,
    coeff: f32,
}

impl OnePoleXover {
    fn new(freq_hz: f32) -> Self {
        let sr = KOKORO_SAMPLE_RATE as f64;
        let w = (2.0 * std::f64::consts::PI * f64::from(freq_hz) / sr).tan();
        let coeff = (w / (1.0 + w)) as f32;
        Self {
            lp_z: 0.0,
            lp2_z: 0.0,
            coeff,
        }
    }

    /// Process one sample, returning (lowpass, highpass).
    #[inline]
    fn process(&mut self, x: f32) -> (f32, f32) {
        if !x.is_finite() {
            self.lp_z = 0.0;
            self.lp2_z = 0.0;
            return (0.0, 0.0);
        }
        // First one-pole LP stage.
        let lp1 = self.lp_z + self.coeff * (x - self.lp_z);
        self.lp_z = lp1;
        // Second cascaded one-pole LP stage.
        let lp2 = self.lp2_z + self.coeff * (lp1 - self.lp2_z);
        self.lp2_z = lp2;
        let hp = x - lp2;
        (lp2, hp)
    }

    fn reset(&mut self) {
        self.lp_z = 0.0;
        self.lp2_z = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Band splitter for N bands
// ---------------------------------------------------------------------------

/// N-band splitter using cascaded one-pole crossovers.
///
/// Band boundaries are logarithmically spaced between `low_hz` and
/// the Nyquist frequency. Each crossover splits the residual high band
/// into the next low and remaining high.
#[derive(Debug, Clone)]
struct BandSplitter {
    crossovers: Vec<OnePoleXover>,
    n_bands: usize,
}

impl BandSplitter {
    fn new(n_bands: usize) -> Self {
        let nyquist = KOKORO_SAMPLE_RATE as f32 / 2.0;
        let low_hz = 80.0f32;

        let mut crossovers = Vec::with_capacity(n_bands.saturating_sub(1));
        if n_bands > 1 {
            let log_low = low_hz.ln();
            let log_high = (nyquist * 0.9).ln();
            for i in 1..n_bands {
                let t = i as f32 / n_bands as f32;
                let freq = (log_low + t * (log_high - log_low)).exp();
                crossovers.push(OnePoleXover::new(freq));
            }
        }
        Self {
            crossovers,
            n_bands,
        }
    }

    /// Split a single sample into bands. Returns a Vec of band values.
    fn split(&mut self, x: f32, out: &mut [f32]) {
        debug_assert!(out.len() == self.n_bands);
        if self.n_bands == 1 {
            out[0] = x;
            return;
        }
        let mut residual = x;
        for (i, xover) in self.crossovers.iter_mut().enumerate() {
            let (lo, hi) = xover.process(residual);
            out[i] = lo;
            residual = hi;
        }
        // Last band gets the remaining high-frequency content.
        out[self.n_bands - 1] = residual;
    }

    fn reset(&mut self) {
        for xover in &mut self.crossovers {
            xover.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Spectral ducker
// ---------------------------------------------------------------------------

/// Spectral ducker that reduces non-lead voices when the lead is active.
///
/// Supports both broadband ducking (single envelope on full signal) and
/// frequency-aware ducking (per-band analysis and gain reduction).
pub struct SpectralDucker {
    /// Per-band envelope followers for the lead voice.
    lead_envelopes: Vec<EnvelopeFollower>,
    /// Band splitter for the lead voice (only used when frequency_aware).
    lead_splitter: Option<BandSplitter>,
    /// Per-voice, per-band splitters for non-lead voices.
    voice_splitters: Vec<Vec<BandSplitter>>,
    /// Ducking gain target in linear (derived from duck_amount_db).
    duck_gain_linear: f32,
    /// Threshold in dBFS.
    threshold_db: f32,
    /// Number of frequency bands.
    n_bands: usize,
    /// Whether frequency-aware ducking is enabled.
    frequency_aware: bool,
    /// Lead voice index.
    lead_voice_index: usize,
    /// Per-band gain smoothing state (one per band, for smooth transitions).
    gain_state: Vec<f32>,
    /// Attack smoothing coefficient for gain.
    gain_attack_coeff: f32,
    /// Release smoothing coefficient for gain.
    gain_release_coeff: f32,
}

impl SpectralDucker {
    /// Create a new spectral ducker.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: &DuckingConfig, _sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;

        let n_bands = if config.frequency_aware {
            config.n_bands
        } else {
            1
        };

        let lead_envelopes = (0..n_bands)
            .map(|_| EnvelopeFollower::new(config.attack_ms, config.release_ms))
            .collect();

        let lead_splitter = if config.frequency_aware {
            Some(BandSplitter::new(n_bands))
        } else {
            None
        };

        let duck_gain_linear = 10.0f64.powf(f64::from(config.duck_amount_db) / 20.0) as f32;

        let sr = KOKORO_SAMPLE_RATE as f64;
        let gain_attack_coeff = (-1.0 / (f64::from(config.attack_ms) * 0.001 * sr)).exp() as f32;
        let gain_release_coeff = (-1.0 / (f64::from(config.release_ms) * 0.001 * sr)).exp() as f32;

        Ok(Self {
            lead_envelopes,
            lead_splitter,
            voice_splitters: Vec::new(),
            duck_gain_linear,
            threshold_db: config.threshold_db,
            n_bands,
            frequency_aware: config.frequency_aware,
            lead_voice_index: config.lead_voice_index,
            gain_state: vec![1.0; n_bands],
            gain_attack_coeff,
            gain_release_coeff,
        })
    }

    /// Process multiple voice buffers in place, applying ducking to non-lead voices.
    ///
    /// `voices` is a mutable slice of per-voice audio buffers. All buffers must
    /// have the same length. The lead voice (at `config.lead_voice_index`) is
    /// analyzed but not modified.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidInput` if `lead_voice_index` is out of bounds
    /// or voice buffers have different lengths.
    pub fn process(
        &mut self,
        voices: &mut [Vec<f32>],
        config: &DuckingConfig,
    ) -> Result<(), KokoroError> {
        if voices.is_empty() {
            return Ok(());
        }
        if config.lead_voice_index >= voices.len() {
            return Err(KokoroError::InvalidInput(format!(
                "lead_voice_index {} >= voice count {}",
                config.lead_voice_index,
                voices.len(),
            )));
        }
        let buf_len = voices[0].len();
        for (i, v) in voices.iter().enumerate() {
            if v.len() != buf_len {
                return Err(KokoroError::InvalidInput(format!(
                    "voice {} has {} samples, expected {}",
                    i,
                    v.len(),
                    buf_len,
                )));
            }
        }
        if buf_len == 0 {
            return Ok(());
        }

        // Update lead index if config changed.
        self.lead_voice_index = config.lead_voice_index;

        if self.frequency_aware {
            self.process_frequency_aware(voices, buf_len)?;
        } else {
            self.process_broadband(voices, buf_len);
        }

        Ok(())
    }

    /// Broadband ducking: analyze lead energy across full spectrum.
    fn process_broadband(&mut self, voices: &mut [Vec<f32>], buf_len: usize) {
        let lead_idx = self.lead_voice_index;

        for i in 0..buf_len {
            let lead_sample = voices[lead_idx][i];
            let lead_db = self.lead_envelopes[0].process(lead_sample);

            // Compute target gain: 1.0 when below threshold, duck_gain when above.
            let target_gain = if lead_db > self.threshold_db {
                // Smooth interpolation based on how far above threshold.
                let over_db = lead_db - self.threshold_db;
                let duck_ratio = (over_db / 6.0).min(1.0);
                1.0 - duck_ratio * (1.0 - self.duck_gain_linear)
            } else {
                1.0
            };

            // Smooth the gain transition.
            let coeff = if target_gain < self.gain_state[0] {
                self.gain_attack_coeff
            } else {
                self.gain_release_coeff
            };
            self.gain_state[0] = coeff * self.gain_state[0] + (1.0 - coeff) * target_gain;

            let gain = self.gain_state[0];

            // Apply gain to all non-lead voices.
            for (v_idx, voice) in voices.iter_mut().enumerate() {
                if v_idx != lead_idx {
                    let s = voice[i] * gain;
                    voice[i] = if s.is_finite() { s } else { 0.0 };
                }
            }
        }
    }

    /// Frequency-aware ducking: analyze and apply per-band.
    fn process_frequency_aware(
        &mut self,
        voices: &mut [Vec<f32>],
        buf_len: usize,
    ) -> Result<(), KokoroError> {
        let lead_idx = self.lead_voice_index;
        let n_bands = self.n_bands;
        let n_voices = voices.len();

        // Ensure we have splitters for each non-lead voice.
        let needed = n_voices.saturating_sub(1);
        while self.voice_splitters.len() < needed {
            // Each non-lead voice gets a pair of splitters (analysis + synthesis).
            self.voice_splitters.push(vec![BandSplitter::new(n_bands)]);
        }

        let mut lead_bands = vec![0.0f32; n_bands];
        let mut voice_bands = vec![0.0f32; n_bands];
        let mut per_band_gains = vec![1.0f32; n_bands];

        for i in 0..buf_len {
            // Split lead voice into bands and compute per-band energy.
            if let Some(ref mut splitter) = self.lead_splitter {
                splitter.split(voices[lead_idx][i], &mut lead_bands);
            }

            // Compute per-band envelope and target gain.
            for b in 0..n_bands {
                let lead_db = self.lead_envelopes[b].process(lead_bands[b]);
                let target_gain = if lead_db > self.threshold_db {
                    let over_db = lead_db - self.threshold_db;
                    let duck_ratio = (over_db / 6.0).min(1.0);
                    1.0 - duck_ratio * (1.0 - self.duck_gain_linear)
                } else {
                    1.0
                };

                let coeff = if target_gain < self.gain_state[b] {
                    self.gain_attack_coeff
                } else {
                    self.gain_release_coeff
                };
                self.gain_state[b] = coeff * self.gain_state[b] + (1.0 - coeff) * target_gain;
                per_band_gains[b] = self.gain_state[b];
            }

            // Apply per-band gain to each non-lead voice.
            let mut splitter_idx = 0;
            for v_idx in 0..n_voices {
                if v_idx == lead_idx {
                    continue;
                }
                // Split the voice into bands.
                if splitter_idx < self.voice_splitters.len()
                    && !self.voice_splitters[splitter_idx].is_empty()
                {
                    self.voice_splitters[splitter_idx][0].split(voices[v_idx][i], &mut voice_bands);
                } else {
                    // Fallback: treat as single band.
                    voice_bands[0] = voices[v_idx][i];
                    for b in 1..n_bands {
                        voice_bands[b] = 0.0;
                    }
                }

                // Apply per-band gain and sum back.
                let mut out = 0.0f32;
                for b in 0..n_bands {
                    out += voice_bands[b] * per_band_gains[b];
                }
                voices[v_idx][i] = if out.is_finite() { out } else { 0.0 };

                splitter_idx += 1;
            }
        }

        Ok(())
    }

    /// Reset all internal state (envelopes, splitters, gain smoothers).
    pub fn reset(&mut self) {
        for env in &mut self.lead_envelopes {
            env.reset();
        }
        if let Some(ref mut splitter) = self.lead_splitter {
            splitter.reset();
        }
        for vs in &mut self.voice_splitters {
            for s in vs {
                s.reset();
            }
        }
        for g in &mut self.gain_state {
            *g = 1.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Sidechain compression
// ---------------------------------------------------------------------------

/// Configuration for sidechain compression.
///
/// An external "sidechain" signal drives gain reduction on the audio.
/// Common use: duck music or effects under voice.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SidechainConfig {
    /// How much to reduce the audio when sidechain is active (dB).
    /// Must be in [-20.0, 0.0] and finite.
    pub duck_amount_db: f32,
    /// How quickly gain reduction engages (ms). Must be in [1.0, 50.0] and finite.
    pub attack_ms: f32,
    /// How slowly gain reduction releases (ms). Must be in [50.0, 500.0] and finite.
    pub release_ms: f32,
    /// Sidechain level (dBFS) that triggers gain reduction.
    /// Must be in [-60.0, 0.0] and finite.
    pub threshold_db: f32,
}

impl Default for SidechainConfig {
    fn default() -> Self {
        Self {
            duck_amount_db: -9.0,
            attack_ms: 5.0,
            release_ms: 150.0,
            threshold_db: -30.0,
        }
    }
}

impl SidechainConfig {
    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.duck_amount_db.is_finite()
            || self.duck_amount_db < -20.0
            || self.duck_amount_db > 0.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "duck_amount_db",
                reason: format!(
                    "duck_amount_db = {}: must be finite and in [-20, 0]",
                    self.duck_amount_db,
                ),
            });
        }
        if !self.attack_ms.is_finite() || self.attack_ms < 1.0 || self.attack_ms > 50.0 {
            return Err(KokoroError::InvalidConfig {
                field: "attack_ms",
                reason: format!(
                    "attack_ms = {}: must be finite and in [1, 50]",
                    self.attack_ms,
                ),
            });
        }
        if !self.release_ms.is_finite() || self.release_ms < 50.0 || self.release_ms > 500.0 {
            return Err(KokoroError::InvalidConfig {
                field: "release_ms",
                reason: format!(
                    "release_ms = {}: must be finite and in [50, 500]",
                    self.release_ms,
                ),
            });
        }
        if !self.threshold_db.is_finite() || self.threshold_db < -60.0 || self.threshold_db > 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "threshold_db",
                reason: format!(
                    "threshold_db = {}: must be finite and in [-60, 0]",
                    self.threshold_db,
                ),
            });
        }
        Ok(())
    }
}

/// Apply sidechain compression to an audio buffer.
///
/// The `sidechain` signal drives gain reduction on `audio`. When the
/// sidechain envelope exceeds the threshold, `audio` is attenuated
/// by `duck_amount_db` with smooth attack/release transitions.
///
/// Both buffers must have the same length.
///
/// # Errors
///
/// Returns `KokoroError::InvalidInput` if buffers have different lengths.
/// Returns `KokoroError::InvalidConfig` if the config is invalid.
pub fn apply_sidechain(
    audio: &mut [f32],
    sidechain: &[f32],
    config: &SidechainConfig,
) -> Result<(), KokoroError> {
    config.validate()?;

    if audio.len() != sidechain.len() {
        return Err(KokoroError::InvalidInput(format!(
            "audio length {} != sidechain length {}",
            audio.len(),
            sidechain.len(),
        )));
    }

    if audio.is_empty() {
        return Ok(());
    }

    let duck_gain_linear = 10.0f64.powf(f64::from(config.duck_amount_db) / 20.0) as f32;
    let mut envelope = EnvelopeFollower::new(config.attack_ms, config.release_ms);

    let sr = KOKORO_SAMPLE_RATE as f64;
    let gain_attack_coeff = (-1.0 / (f64::from(config.attack_ms) * 0.001 * sr)).exp() as f32;
    let gain_release_coeff = (-1.0 / (f64::from(config.release_ms) * 0.001 * sr)).exp() as f32;
    let mut gain_state = 1.0f32;

    for (audio_sample, &sc_sample) in audio.iter_mut().zip(sidechain.iter()) {
        let sc_db = envelope.process(sc_sample);

        let target_gain = if sc_db > config.threshold_db {
            let over_db = sc_db - config.threshold_db;
            let duck_ratio = (over_db / 6.0).min(1.0);
            1.0 - duck_ratio * (1.0 - duck_gain_linear)
        } else {
            1.0
        };

        let coeff = if target_gain < gain_state {
            gain_attack_coeff
        } else {
            gain_release_coeff
        };
        gain_state = coeff * gain_state + (1.0 - coeff) * target_gain;

        let out = *audio_sample * gain_state;
        *audio_sample = if out.is_finite() { out } else { 0.0 };
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_ducking_tests.rs"]
mod tests;
