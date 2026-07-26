// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Breathing patterns and micro-timing humanization for multi-voice chorus.
//!
//! Real choirs sound human because singers breathe at different times, have
//! slight timing variations, and have natural amplitude envelopes (attacks,
//! sustains, releases). This module adds those human elements to synthesized
//! multi-voice output.
//!
//! # Design
//!
//! All randomization is **deterministic** given a seed derived from the voice
//! index. This means identical inputs produce identical outputs across runs,
//! which is critical for reproducible audio quality testing and verification.
//!
//! The humanization is **subtle by design**: breathing creates gentle amplitude
//! dips (not silence), micro-timing adds slight onset jitter (not rhythmic
//! distortion), and amplitude envelopes shape attacks/releases naturally.
//!
//! # Usage
//!
//! ```ignore
//! let config = HumanizeConfig::default();
//! let mut audio = vec![0.5f32; 48000]; // 2 seconds at 24kHz
//! apply_humanize(&mut audio, &config, 0, 24000);
//! ```

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Deterministic PRNG (LCG)
// ---------------------------------------------------------------------------

/// Simple linear congruential generator for deterministic pseudo-random values.
///
/// Uses the Numerical Recipes LCG parameters. Fast, small state, and fully
/// deterministic given the same seed -- exactly what we need for per-voice
/// reproducible humanization.
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Create a new LCG seeded from a voice index and a domain salt.
    fn new(voice_index: usize, salt: u64) -> Self {
        // Mix voice_index with salt to create diverse per-domain seeds.
        let seed = (voice_index as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(salt)
            .wrapping_add(1);
        Self { state: seed }
    }

    /// Return the next pseudo-random u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Return a pseudo-random f32 in [0.0, 1.0).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Return a pseudo-random f32 in [lo, hi).
    #[inline]
    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

// ---------------------------------------------------------------------------
// Breath pattern
// ---------------------------------------------------------------------------

/// Configuration for natural breathing patterns applied to a voice.
///
/// Simulates the amplitude dips that occur when a singer breathes. In a real
/// choir, singers stagger their breaths so the overall level stays consistent.
/// Each voice gets independent breath timing derived from its seed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BreathPattern {
    /// Minimum interval between breaths in seconds.
    ///
    /// Range: [1.0, 10.0]. Default: `2.0`.
    pub min_interval_sec: f32,

    /// Maximum interval between breaths in seconds.
    ///
    /// Range: [1.0, 10.0], must be >= `min_interval_sec`. Default: `6.0`.
    pub max_interval_sec: f32,

    /// Duration of each breath dip in seconds.
    ///
    /// Range: [0.05, 0.5]. Default: `0.15`.
    pub breath_duration_sec: f32,

    /// Depth of the amplitude reduction during a breath (0.0 = no dip, 1.0 = silence).
    ///
    /// Range: [0.0, 1.0]. Default: `0.25`. Typical singing breaths reduce
    /// amplitude by 15-35%, not to silence.
    pub breath_depth: f32,
}

impl Default for BreathPattern {
    fn default() -> Self {
        Self {
            min_interval_sec: 2.0,
            max_interval_sec: 6.0,
            breath_duration_sec: 0.15,
            breath_depth: 0.25,
        }
    }
}

impl BreathPattern {
    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.min_interval_sec.is_finite() || !(1.0..=10.0).contains(&self.min_interval_sec) {
            return Err(KokoroError::InvalidConfig {
                field: "min_interval_sec",
                reason: format!(
                    "min_interval_sec = {}: must be finite and in [1.0, 10.0]",
                    self.min_interval_sec,
                ),
            });
        }
        if !self.max_interval_sec.is_finite() || !(1.0..=10.0).contains(&self.max_interval_sec) {
            return Err(KokoroError::InvalidConfig {
                field: "max_interval_sec",
                reason: format!(
                    "max_interval_sec = {}: must be finite and in [1.0, 10.0]",
                    self.max_interval_sec,
                ),
            });
        }
        if self.min_interval_sec > self.max_interval_sec {
            return Err(KokoroError::InvalidConfig {
                field: "max_interval_sec",
                reason: format!(
                    "max_interval_sec ({}) must be >= min_interval_sec ({})",
                    self.max_interval_sec, self.min_interval_sec,
                ),
            });
        }
        if !self.breath_duration_sec.is_finite()
            || !(0.05..=0.5).contains(&self.breath_duration_sec)
        {
            return Err(KokoroError::InvalidConfig {
                field: "breath_duration_sec",
                reason: format!(
                    "breath_duration_sec = {}: must be finite and in [0.05, 0.5]",
                    self.breath_duration_sec,
                ),
            });
        }
        if !self.breath_depth.is_finite() || !(0.0..=1.0).contains(&self.breath_depth) {
            return Err(KokoroError::InvalidConfig {
                field: "breath_depth",
                reason: format!(
                    "breath_depth = {}: must be finite and in [0.0, 1.0]",
                    self.breath_depth,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Micro-timing
// ---------------------------------------------------------------------------

/// Configuration for micro-timing humanization (onset jitter and tempo drift).
///
/// Real singers never hit note onsets at exactly the same time. Small timing
/// variations (on the order of a few milliseconds) make a chorus sound alive
/// rather than robotic.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MicroTiming {
    /// Maximum onset jitter per phrase boundary in seconds.
    ///
    /// Range: [0.0, 0.030]. Default: `0.008` (8ms). Applied as random
    /// per-phrase sample shifts. Larger values sound more "loose."
    pub onset_jitter_sec: f32,

    /// Maximum tempo drift as a fraction (0.0 = none, 0.02 = +/-2%).
    ///
    /// Range: [0.0, 0.05]. Default: `0.005` (0.5%). Applied as a slowly
    /// varying time-stretch. Larger values sound more human but risk
    /// audible pitch artifacts at >2%.
    pub tempo_drift_max: f32,

    /// Rate of tempo drift variation in Hz (how fast the drift changes).
    ///
    /// Range: [0.05, 2.0]. Default: `0.3`. Lower = slower wander, higher =
    /// more restless. Natural singing tends toward 0.2-0.5 Hz.
    pub drift_rate_hz: f32,
}

impl Default for MicroTiming {
    fn default() -> Self {
        Self {
            onset_jitter_sec: 0.008,
            tempo_drift_max: 0.005,
            drift_rate_hz: 0.3,
        }
    }
}

impl MicroTiming {
    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.onset_jitter_sec.is_finite() || !(0.0..=0.030).contains(&self.onset_jitter_sec) {
            return Err(KokoroError::InvalidConfig {
                field: "onset_jitter_sec",
                reason: format!(
                    "onset_jitter_sec = {}: must be finite and in [0.0, 0.030]",
                    self.onset_jitter_sec,
                ),
            });
        }
        if !self.tempo_drift_max.is_finite() || !(0.0..=0.05).contains(&self.tempo_drift_max) {
            return Err(KokoroError::InvalidConfig {
                field: "tempo_drift_max",
                reason: format!(
                    "tempo_drift_max = {}: must be finite and in [0.0, 0.05]",
                    self.tempo_drift_max,
                ),
            });
        }
        if !self.drift_rate_hz.is_finite() || !(0.05..=2.0).contains(&self.drift_rate_hz) {
            return Err(KokoroError::InvalidConfig {
                field: "drift_rate_hz",
                reason: format!(
                    "drift_rate_hz = {}: must be finite and in [0.05, 2.0]",
                    self.drift_rate_hz,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Amplitude envelope (AHDSR)
// ---------------------------------------------------------------------------

/// Amplitude envelope applied to each voice for natural attack/release shaping.
///
/// Without an envelope, voices appear and disappear abruptly. This shapes the
/// onset (attack), steady state (hold + sustain), and ending (release) to
/// mimic how a real singer enters and exits a phrase.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AmplitudeEnvelope {
    /// Attack time in seconds (ramp from 0 to 1).
    ///
    /// Range: [0.001, 0.200]. Default: `0.025` (25ms).
    pub attack_sec: f32,

    /// Hold time at full amplitude in seconds (after attack, before decay).
    ///
    /// Range: [0.0, 1.0]. Default: `0.0` (no hold, go straight to decay).
    pub hold_sec: f32,

    /// Decay time in seconds (ramp from 1 to sustain level).
    ///
    /// Range: [0.0, 1.0]. Default: `0.050` (50ms).
    pub decay_sec: f32,

    /// Sustain level (0.0 to 1.0) held during the body of the audio.
    ///
    /// Range: [0.0, 1.0]. Default: `1.0` (no decay, full amplitude).
    pub sustain_level: f32,

    /// Release time in seconds (ramp from sustain to 0 at the end).
    ///
    /// Range: [0.005, 0.500]. Default: `0.080` (80ms).
    pub release_sec: f32,
}

impl Default for AmplitudeEnvelope {
    fn default() -> Self {
        Self {
            attack_sec: 0.025,
            hold_sec: 0.0,
            decay_sec: 0.050,
            sustain_level: 1.0,
            release_sec: 0.080,
        }
    }
}

impl AmplitudeEnvelope {
    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.attack_sec.is_finite() || !(0.001..=0.200).contains(&self.attack_sec) {
            return Err(KokoroError::InvalidConfig {
                field: "attack_sec",
                reason: format!(
                    "attack_sec = {}: must be finite and in [0.001, 0.200]",
                    self.attack_sec,
                ),
            });
        }
        if !self.hold_sec.is_finite() || !(0.0..=1.0).contains(&self.hold_sec) {
            return Err(KokoroError::InvalidConfig {
                field: "hold_sec",
                reason: format!(
                    "hold_sec = {}: must be finite and in [0.0, 1.0]",
                    self.hold_sec,
                ),
            });
        }
        if !self.decay_sec.is_finite() || !(0.0..=1.0).contains(&self.decay_sec) {
            return Err(KokoroError::InvalidConfig {
                field: "decay_sec",
                reason: format!(
                    "decay_sec = {}: must be finite and in [0.0, 1.0]",
                    self.decay_sec,
                ),
            });
        }
        if !self.sustain_level.is_finite() || !(0.0..=1.0).contains(&self.sustain_level) {
            return Err(KokoroError::InvalidConfig {
                field: "sustain_level",
                reason: format!(
                    "sustain_level = {}: must be finite and in [0.0, 1.0]",
                    self.sustain_level,
                ),
            });
        }
        if !self.release_sec.is_finite() || !(0.005..=0.500).contains(&self.release_sec) {
            return Err(KokoroError::InvalidConfig {
                field: "release_sec",
                reason: format!(
                    "release_sec = {}: must be finite and in [0.005, 0.500]",
                    self.release_sec,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Combined humanization config
// ---------------------------------------------------------------------------

/// Combined humanization configuration for a chorus voice.
///
/// Bundles breathing, micro-timing, and amplitude envelope into a single
/// config that can be applied to each voice's PCM audio before mixing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HumanizeConfig {
    /// Breathing pattern (amplitude dips at natural intervals).
    pub breath: BreathPattern,

    /// Micro-timing (onset jitter and tempo drift).
    pub timing: MicroTiming,

    /// Amplitude envelope (attack/release shaping).
    pub envelope: AmplitudeEnvelope,

    /// Enable breathing humanization. Default: `true`.
    pub enable_breath: bool,

    /// Enable micro-timing humanization. Default: `true`.
    pub enable_timing: bool,

    /// Enable amplitude envelope. Default: `true`.
    pub enable_envelope: bool,
}

impl Default for HumanizeConfig {
    fn default() -> Self {
        Self {
            breath: BreathPattern::default(),
            timing: MicroTiming::default(),
            envelope: AmplitudeEnvelope::default(),
            enable_breath: true,
            enable_timing: true,
            enable_envelope: true,
        }
    }
}

impl HumanizeConfig {
    /// Create a humanize config with all effects disabled (pass-through).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enable_breath: false,
            enable_timing: false,
            enable_envelope: false,
            ..Self::default()
        }
    }

    /// Enable only breathing humanization.
    #[must_use]
    pub fn breath_only() -> Self {
        Self {
            enable_breath: true,
            enable_timing: false,
            enable_envelope: false,
            ..Self::default()
        }
    }

    /// Set a custom breathing pattern.
    #[must_use]
    pub fn with_breath(mut self, breath: BreathPattern) -> Self {
        self.breath = breath;
        self
    }

    /// Set a custom micro-timing config.
    #[must_use]
    pub fn with_timing(mut self, timing: MicroTiming) -> Self {
        self.timing = timing;
        self
    }

    /// Set a custom amplitude envelope.
    #[must_use]
    pub fn with_envelope(mut self, envelope: AmplitudeEnvelope) -> Self {
        self.envelope = envelope;
        self
    }

    /// Validate all sub-configs.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.enable_breath {
            self.breath.validate()?;
        }
        if self.enable_timing {
            self.timing.validate()?;
        }
        if self.enable_envelope {
            self.envelope.validate()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Breath pattern application
// ---------------------------------------------------------------------------

/// Apply breathing dips to audio at pseudo-random intervals.
///
/// Breathing dips are smooth (raised-cosine shaped) amplitude reductions
/// placed at stochastic intervals. The shape avoids abrupt transitions
/// that would sound like glitches.
fn apply_breath_pattern(
    audio: &mut [f32],
    breath: &BreathPattern,
    voice_index: usize,
    sample_rate: u32,
) {
    let sr = sample_rate as f32;
    let breath_samples = (breath.breath_duration_sec * sr).round() as usize;
    if breath_samples == 0 || audio.is_empty() {
        return;
    }

    let mut rng = Lcg::new(voice_index, 0xB4EA_7777_CAFE_0001);

    // Walk through the audio placing breath dips at random intervals.
    let mut pos: usize = 0;
    loop {
        // Random interval until next breath.
        let interval_sec = rng.next_f32_range(breath.min_interval_sec, breath.max_interval_sec);
        let interval_samples = (interval_sec * sr).round() as usize;
        pos += interval_samples;

        if pos + breath_samples > audio.len() {
            break;
        }

        // Apply a raised-cosine dip centered at `pos`.
        for i in 0..breath_samples {
            let t = i as f32 / breath_samples as f32;
            // Raised cosine: 1.0 at edges, dips to (1-depth) at center.
            let cos_val = (t * std::f32::consts::PI * 2.0).cos();
            let envelope = 1.0 - breath.breath_depth * 0.5 * (1.0 - cos_val);
            audio[pos + i] *= envelope;
        }
    }
}

// ---------------------------------------------------------------------------
// Micro-timing application (tempo drift)
// ---------------------------------------------------------------------------

/// Apply micro-timing drift to audio via slow sinusoidal time warping.
///
/// This creates a subtle time-varying stretch/compress effect that mimics
/// a singer drifting slightly ahead or behind the beat. The drift is a
/// low-frequency sinusoid so it changes smoothly.
fn apply_micro_timing(
    audio: &mut [f32],
    timing: &MicroTiming,
    voice_index: usize,
    sample_rate: u32,
) {
    if audio.is_empty() || timing.tempo_drift_max < 1e-7 {
        return;
    }

    let sr = sample_rate as f32;
    let len = audio.len();

    // Per-voice phase offset so each voice drifts independently.
    let mut rng = Lcg::new(voice_index, 0xD1F7_CAFE_5EED_0002);
    let phase_offset = rng.next_f32() * std::f32::consts::TAU;

    // Build a time-warped version of the audio. For each output sample,
    // compute the source position with a sinusoidal drift.
    let mut warped = vec![0.0f32; len];
    for i in 0..len {
        let t = i as f32 / sr;
        // Sinusoidal drift: the cumulative time offset oscillates.
        let drift = timing.tempo_drift_max
            * (std::f32::consts::TAU * timing.drift_rate_hz * t + phase_offset).sin();
        let src_pos = i as f64 + (f64::from(drift) * f64::from(sr));
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        if src_idx < len {
            let s0 = audio[src_idx];
            let s1 = if src_idx + 1 < len {
                audio[src_idx + 1]
            } else {
                s0
            };
            warped[i] = s0 + frac * (s1 - s0);
        }
        // Else: out-of-bounds source stays zero (natural fade).
    }

    audio.copy_from_slice(&warped);
}

// ---------------------------------------------------------------------------
// Amplitude envelope application
// ---------------------------------------------------------------------------

/// Apply AHDSR amplitude envelope to audio.
///
/// Shapes the beginning (attack + hold + decay) and end (release) of the
/// audio to mimic natural vocal onset and offset.
fn apply_amplitude_envelope(audio: &mut [f32], env: &AmplitudeEnvelope, sample_rate: u32) {
    if audio.is_empty() {
        return;
    }

    let sr = sample_rate as f32;
    let len = audio.len();

    let attack_samples = (env.attack_sec * sr).round() as usize;
    let hold_samples = (env.hold_sec * sr).round() as usize;
    let decay_samples = (env.decay_sec * sr).round() as usize;
    let release_samples = (env.release_sec * sr).round() as usize;

    // Compute envelope gain at each sample position.
    let onset_total = attack_samples + hold_samples + decay_samples;

    for (i, sample) in audio.iter_mut().enumerate() {
        let gain = if i < attack_samples {
            // Attack: linear ramp 0 -> 1.
            if attack_samples > 0 {
                i as f32 / attack_samples as f32
            } else {
                1.0
            }
        } else if i < attack_samples + hold_samples {
            // Hold: stay at 1.0.
            1.0
        } else if i < onset_total {
            // Decay: linear ramp 1 -> sustain_level.
            if decay_samples > 0 {
                let decay_pos = (i - attack_samples - hold_samples) as f32;
                1.0 - (1.0 - env.sustain_level) * (decay_pos / decay_samples as f32)
            } else {
                env.sustain_level
            }
        } else if len > release_samples && i >= len - release_samples {
            // Release: linear ramp sustain_level -> 0.
            let release_pos = (i - (len - release_samples)) as f32;
            env.sustain_level * (1.0 - release_pos / release_samples as f32)
        } else {
            // Sustain.
            env.sustain_level
        };

        *sample *= gain;
    }
}

// ---------------------------------------------------------------------------
// Onset jitter (circular shift)
// ---------------------------------------------------------------------------

/// Apply onset jitter as a circular shift of the audio buffer.
///
/// Shifts the audio by a voice-dependent number of samples derived from the
/// voice index and the configured `onset_jitter_sec`. This simulates
/// singers starting at slightly different times -- a key element of natural
/// chorus sound.
///
/// The shift amount is deterministic per voice index.
fn apply_onset_jitter(
    audio: &mut [f32],
    timing: &MicroTiming,
    voice_index: usize,
    sample_rate: u32,
) {
    if audio.is_empty() || timing.onset_jitter_sec < 1e-7 {
        return;
    }

    let max_jitter_samples = (timing.onset_jitter_sec * sample_rate as f32).round() as usize;
    if max_jitter_samples == 0 {
        return;
    }

    // Deterministic jitter amount per voice.
    let mut rng = Lcg::new(voice_index, 0x0115_E7A1_77E4_0003);
    let jitter_samples = (rng.next_f32() * max_jitter_samples as f32).round() as usize;
    if jitter_samples == 0 || jitter_samples >= audio.len() {
        return;
    }

    // Circular shift right by `jitter_samples`.
    let len = audio.len();
    let mut buf = vec![0.0f32; len];
    buf[..jitter_samples].copy_from_slice(&audio[len - jitter_samples..]);
    buf[jitter_samples..].copy_from_slice(&audio[..len - jitter_samples]);
    audio.copy_from_slice(&buf);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply a circular-shift jitter to audio, returning a new buffer.
///
/// Shifts audio right by `jitter_samples` positions (wrapping around).
/// Each voice should get a different offset to simulate onset timing
/// differences.
///
/// # Returns
///
/// A new `Vec<f32>` of the same length with the shifted audio.
#[must_use]
pub fn apply_jitter(audio: &[f32], jitter_samples: usize, _voice_index: usize) -> Vec<f32> {
    let len = audio.len();
    if len == 0 || jitter_samples == 0 {
        return audio.to_vec();
    }
    let shift = jitter_samples % len;
    if shift == 0 {
        return audio.to_vec();
    }
    let mut out = vec![0.0f32; len];
    out[..shift].copy_from_slice(&audio[len - shift..]);
    out[shift..].copy_from_slice(&audio[..len - shift]);
    out
}

/// Apply humanization effects to a single voice's PCM audio.
///
/// Applies breathing patterns, micro-timing drift, onset jitter, and
/// amplitude envelope shaping based on the provided configuration. Each
/// effect is independently toggleable via the `enable_*` flags on
/// [`HumanizeConfig`].
///
/// The `voice_index` seeds all random elements, ensuring deterministic output
/// for the same index and different humanization per voice.
///
/// # Arguments
///
/// * `voice_audio` - Mutable PCM audio buffer (modified in-place).
/// * `config` - Humanization configuration.
/// * `voice_index` - Index of this voice (seeds per-voice randomness).
/// * `sample_rate` - Audio sample rate in Hz (typically 24000 for Kokoro).
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if any config parameter is out of range.
pub fn apply_humanize(
    voice_audio: &mut [f32],
    config: &HumanizeConfig,
    voice_index: usize,
    sample_rate: u32,
) -> Result<(), KokoroError> {
    config.validate()?;

    if voice_audio.is_empty() {
        return Ok(());
    }

    // Order: envelope first (shapes onset/offset), then breathing (dips),
    // then timing drift (warps), then onset jitter (circular shift).
    if config.enable_envelope {
        apply_amplitude_envelope(voice_audio, &config.envelope, sample_rate);
    }
    if config.enable_breath {
        apply_breath_pattern(voice_audio, &config.breath, voice_index, sample_rate);
    }
    if config.enable_timing {
        apply_micro_timing(voice_audio, &config.timing, voice_index, sample_rate);
        apply_onset_jitter(voice_audio, &config.timing, voice_index, sample_rate);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_humanize_tests.rs"]
mod tests;
