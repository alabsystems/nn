// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Breath and pause modeling for multi-voice chorus naturalness.
//!
//! Real choirs breathe at different times. When all synthesized voices pause
//! simultaneously, the silence is unnatural and breaks the illusion of a live
//! ensemble. This module detects pauses in generated audio, synthesizes subtle
//! breath noise, and inserts it with per-voice timing stagger so each voice
//! "breathes" independently.
//!
//! # Design
//!
//! Breath sounds are shaped pink noise filtered to the vocal frequency range
//! (~200-2000 Hz) using a simple one-pole lowpass filter. Each voice gets a
//! deterministic timing offset derived from its index, so breath placement
//! differs across voices while remaining reproducible across runs.
//!
//! Pause detection uses a windowed energy measurement: regions where the
//! RMS energy falls below a configurable threshold are identified as pauses.
//! Breath sounds are inserted at the start of each detected pause with a
//! smooth fade-in/fade-out envelope.
//!
//! # Placement in the chorus pipeline
//!
//! Breath insertion is applied **after** humanization and **before** final
//! stereo mixing:
//! ```text
//! Per-voice: vibrato -> detuning -> EQ -> humanize -> breath -> stereo mix
//! ```
//!
//! This ordering is deliberate: humanization shapes the voice's amplitude
//! envelope and breathing dips first, then breath noise fills the resulting
//! quiet regions. Inserting breath before humanization would fight with the
//! breath pattern amplitude dips.
//!
//! # Usage
//!
//! ```ignore
//! let config = BreathConfig::default();
//! let mut generator = BreathGenerator::new(&config, 4).unwrap();
//! let pauses = detect_pauses(&audio, &config);
//! let mut voices = vec![audio.clone(); 4];
//! insert_breath_sounds(&mut voices, &pauses, &mut generator, &config).unwrap();
//! ```

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Deterministic PRNG (LCG)
// ---------------------------------------------------------------------------

/// Simple linear congruential generator for deterministic pseudo-random values.
///
/// Uses the Numerical Recipes LCG parameters. Fast, small state, and fully
/// deterministic given the same seed.
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Create a new LCG seeded from a voice index and a domain salt.
    fn new(voice_index: usize, salt: u64) -> Self {
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

    /// Return a pseudo-random f32 in [-1.0, 1.0).
    #[inline]
    fn next_f32_bipolar(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for breath noise and pause detection.
///
/// Controls the volume, duration, and timing stagger of synthetic breath
/// sounds inserted at detected pauses in multi-voice chorus audio.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BreathConfig {
    /// Volume of breath noise relative to full scale.
    ///
    /// Range: [0.0, 0.2]. Default: `0.02`. Breath sounds should be barely
    /// audible -- just enough to fill silence naturally.
    pub breath_noise_level: f32,

    /// Typical duration of a single breath sound in milliseconds.
    ///
    /// Range: [20.0, 500.0]. Default: `80.0`. Real vocal breaths are
    /// 50-200ms; shorter values sound like clicks, longer like sighs.
    pub breath_duration_ms: f32,

    /// Maximum timing stagger between voices in milliseconds.
    ///
    /// Range: [0.0, 200.0]. Default: `30.0`. Each voice's breath is offset
    /// by a deterministic amount up to this maximum, preventing the
    /// unnatural "simultaneous breath" artifact.
    pub stagger_ms: f32,

    /// Audio sample rate in Hz.
    ///
    /// Must be in [8000.0, 96000.0] and finite. Default: `24000.0` (Kokoro).
    pub sample_rate: f32,

    /// RMS energy threshold for pause detection (linear amplitude).
    ///
    /// Range: [0.0001, 0.5]. Default: `0.01`. Regions with windowed RMS
    /// below this threshold are classified as pauses.
    pub pause_threshold: f32,

    /// Analysis window size for pause detection in milliseconds.
    ///
    /// Range: [5.0, 100.0]. Default: `20.0`. Smaller windows detect
    /// shorter pauses but are noisier; larger windows are smoother but
    /// may miss brief gaps.
    pub pause_window_ms: f32,

    /// One-pole lowpass filter cutoff frequency in Hz for breath shaping.
    ///
    /// Range: [100.0, 5000.0]. Default: `2000.0`. Filters the raw noise
    /// to approximate the spectral shape of a vocal breath (bandlimited
    /// to roughly 200-2000 Hz). Lower values produce a warmer breath
    /// sound; higher values are breathier/airier.
    pub filter_cutoff_hz: f32,
}

impl BreathConfig {
    /// Create a new `BreathConfig` with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set breath noise level.
    #[must_use]
    pub fn with_noise_level(mut self, level: f32) -> Self {
        self.breath_noise_level = level;
        self
    }

    /// Builder: set breath duration in milliseconds.
    #[must_use]
    pub fn with_duration_ms(mut self, ms: f32) -> Self {
        self.breath_duration_ms = ms;
        self
    }

    /// Builder: set maximum timing stagger in milliseconds.
    #[must_use]
    pub fn with_stagger_ms(mut self, ms: f32) -> Self {
        self.stagger_ms = ms;
        self
    }

    /// Builder: set sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, sr: f32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Builder: set pause detection threshold.
    #[must_use]
    pub fn with_pause_threshold(mut self, threshold: f32) -> Self {
        self.pause_threshold = threshold;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.breath_noise_level.is_finite() || !(0.0..=0.2).contains(&self.breath_noise_level) {
            return Err(KokoroError::InvalidConfig {
                field: "breath_noise_level",
                reason: format!(
                    "must be finite and in [0.0, 0.2], got {}",
                    self.breath_noise_level,
                ),
            });
        }
        if !self.breath_duration_ms.is_finite()
            || !(20.0..=500.0).contains(&self.breath_duration_ms)
        {
            return Err(KokoroError::InvalidConfig {
                field: "breath_duration_ms",
                reason: format!(
                    "must be finite and in [20.0, 500.0], got {}",
                    self.breath_duration_ms,
                ),
            });
        }
        if !self.stagger_ms.is_finite() || !(0.0..=200.0).contains(&self.stagger_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "stagger_ms",
                reason: format!(
                    "must be finite and in [0.0, 200.0], got {}",
                    self.stagger_ms,
                ),
            });
        }
        if !self.sample_rate.is_finite() || !(8000.0..=96000.0).contains(&self.sample_rate) {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!(
                    "must be finite and in [8000.0, 96000.0], got {}",
                    self.sample_rate,
                ),
            });
        }
        if !self.pause_threshold.is_finite() || !(0.0001..=0.5).contains(&self.pause_threshold) {
            return Err(KokoroError::InvalidConfig {
                field: "pause_threshold",
                reason: format!(
                    "must be finite and in [0.0001, 0.5], got {}",
                    self.pause_threshold,
                ),
            });
        }
        if !self.pause_window_ms.is_finite() || !(5.0..=100.0).contains(&self.pause_window_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "pause_window_ms",
                reason: format!(
                    "must be finite and in [5.0, 100.0], got {}",
                    self.pause_window_ms,
                ),
            });
        }
        if !self.filter_cutoff_hz.is_finite() || !(100.0..=5000.0).contains(&self.filter_cutoff_hz)
        {
            return Err(KokoroError::InvalidConfig {
                field: "filter_cutoff_hz",
                reason: format!(
                    "must be finite and in [100.0, 5000.0], got {}",
                    self.filter_cutoff_hz,
                ),
            });
        }
        Ok(())
    }
}

impl Default for BreathConfig {
    fn default() -> Self {
        Self {
            breath_noise_level: 0.02,
            breath_duration_ms: 80.0,
            stagger_ms: 30.0,
            sample_rate: 24000.0,
            pause_threshold: 0.01,
            pause_window_ms: 20.0,
            filter_cutoff_hz: 2000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pause detection
// ---------------------------------------------------------------------------

/// A detected pause region in the audio signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PauseRegion {
    /// Start sample index of the pause.
    pub start: usize,
    /// Duration of the pause in samples.
    pub duration: usize,
}

/// Detect pauses (quiet regions) in mono PCM audio.
///
/// Scans the audio using a sliding RMS energy window. Regions where the
/// windowed RMS falls below `config.pause_threshold` for at least one
/// full window are reported as pauses.
///
/// # Arguments
///
/// * `audio` - Mono PCM audio buffer.
/// * `config` - Breath configuration (uses `pause_threshold`, `pause_window_ms`,
///   and `sample_rate`).
///
/// # Returns
///
/// A sorted list of [`PauseRegion`]s, merged so that adjacent or overlapping
/// quiet windows are combined into single regions.
#[must_use]
pub fn detect_pauses(audio: &[f32], config: &BreathConfig) -> Vec<PauseRegion> {
    if audio.is_empty() || !config.sample_rate.is_finite() || config.sample_rate <= 0.0 {
        return Vec::new();
    }

    let window_samples = (config.pause_window_ms * 0.001 * config.sample_rate).round() as usize;
    let window_samples = window_samples.max(1);

    if audio.len() < window_samples {
        return Vec::new();
    }

    let threshold_sq = config.pause_threshold * config.pause_threshold;
    let inv_window = 1.0 / window_samples as f32;

    // Sliding window: compute sum of squares incrementally.
    let mut sum_sq: f64 = audio[..window_samples]
        .iter()
        .map(|&s| {
            let v = if s.is_finite() { s } else { 0.0 };
            f64::from(v) * f64::from(v)
        })
        .sum();

    let mut pauses: Vec<PauseRegion> = Vec::new();
    let mut in_pause = false;
    let mut pause_start = 0usize;

    let n_windows = audio.len() - window_samples + 1;
    for i in 0..n_windows {
        let rms_sq = (sum_sq as f32) * inv_window;

        if rms_sq < threshold_sq {
            if !in_pause {
                in_pause = true;
                pause_start = i;
            }
        } else if in_pause {
            in_pause = false;
            pauses.push(PauseRegion {
                start: pause_start,
                duration: (i - pause_start) + window_samples,
            });
        }

        // Slide window forward.
        if i + window_samples < audio.len() {
            let old = {
                let v = audio[i];
                let v = if v.is_finite() { v } else { 0.0 };
                f64::from(v) * f64::from(v)
            };
            let new = {
                let v = audio[i + window_samples];
                let v = if v.is_finite() { v } else { 0.0 };
                f64::from(v) * f64::from(v)
            };
            sum_sq = (sum_sq - old + new).max(0.0);
        }
    }

    // Close any open pause region at the end.
    if in_pause {
        pauses.push(PauseRegion {
            start: pause_start,
            duration: (n_windows - pause_start) + window_samples - 1,
        });
    }

    pauses
}

// ---------------------------------------------------------------------------
// Breath generator
// ---------------------------------------------------------------------------

/// Generates filtered breath noise for insertion at pauses.
///
/// Each voice gets a deterministic PRNG and one-pole lowpass filter state.
/// Breath sounds are shaped white noise passed through a lowpass filter
/// (approximating pink/vocal breath spectrum) with a smooth fade-in/fade-out
/// envelope.
pub struct BreathGenerator {
    /// Per-voice PRNG state.
    rngs: Vec<Lcg>,
    /// Per-voice one-pole filter state.
    filter_states: Vec<f32>,
    /// One-pole lowpass coefficient (derived from cutoff and sample rate).
    alpha: f32,
    /// Number of voices.
    n_voices: usize,
}

impl BreathGenerator {
    /// Create a new breath generator for `n_voices` voices.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the config is invalid or
    /// `n_voices` is zero.
    pub fn new(config: &BreathConfig, n_voices: usize) -> Result<Self, KokoroError> {
        config.validate()?;

        if n_voices == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: "must be >= 1".to_string(),
            });
        }

        // One-pole lowpass coefficient: alpha = 1 - e^(-2*pi*fc/fs)
        let fc = f64::from(config.filter_cutoff_hz);
        let fs = f64::from(config.sample_rate);
        let alpha = (1.0 - (-std::f64::consts::TAU * fc / fs).exp()) as f32;
        let alpha = if alpha.is_finite() { alpha } else { 0.5 };

        let rngs = (0..n_voices)
            .map(|i| Lcg::new(i, 0xBEEF_CAFE_B4E7_0042))
            .collect();
        let filter_states = vec![0.0f32; n_voices];

        Ok(Self {
            rngs,
            filter_states,
            alpha,
            n_voices,
        })
    }

    /// Generate a breath noise buffer for a single voice.
    ///
    /// Produces `duration_samples` of filtered noise at the configured
    /// level with a raised-cosine fade-in/fade-out envelope.
    ///
    /// # Arguments
    ///
    /// * `voice_index` - Which voice (indexes into internal PRNG/filter state).
    /// * `duration_samples` - Number of samples of breath noise to generate.
    /// * `level` - Peak amplitude of the breath noise.
    pub fn generate(
        &mut self,
        voice_index: usize,
        duration_samples: usize,
        level: f32,
    ) -> Vec<f32> {
        if voice_index >= self.n_voices || duration_samples == 0 {
            return Vec::new();
        }

        let level = if level.is_finite() { level } else { 0.0 };
        let rng = &mut self.rngs[voice_index];
        let filter_state = &mut self.filter_states[voice_index];

        let mut out = Vec::with_capacity(duration_samples);

        // Fade region: 15% of total duration on each side.
        let fade_len = (duration_samples as f32 * 0.15).round() as usize;
        let fade_len = fade_len.max(1).min(duration_samples / 2);

        for i in 0..duration_samples {
            // White noise source.
            let noise = rng.next_f32_bipolar();

            // One-pole lowpass: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
            let filtered = self.alpha * noise + (1.0 - self.alpha) * *filter_state;
            let filtered = if filtered.is_finite() { filtered } else { 0.0 };
            *filter_state = filtered;

            // Raised-cosine fade envelope.
            let envelope = if i < fade_len {
                // Fade in: 0.5 * (1 - cos(pi * t))
                let t = i as f32 / fade_len as f32;
                0.5 * (1.0 - (std::f32::consts::PI * t).cos())
            } else if i >= duration_samples - fade_len {
                // Fade out: 0.5 * (1 + cos(pi * t))
                let t = (i - (duration_samples - fade_len)) as f32 / fade_len as f32;
                0.5 * (1.0 + (std::f32::consts::PI * t).cos())
            } else {
                1.0
            };

            let sample = filtered * level * envelope;
            out.push(if sample.is_finite() { sample } else { 0.0 });
        }

        out
    }

    /// Reset all per-voice filter states (e.g., between segments).
    pub fn reset(&mut self) {
        for s in &mut self.filter_states {
            *s = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Breath insertion
// ---------------------------------------------------------------------------

/// Insert breath sounds at detected pauses across all voices.
///
/// For each detected pause region, generates a breath noise snippet for
/// each voice with a per-voice timing stagger. The breath is mixed (added)
/// into the existing audio at the pause location, so any residual signal
/// in the "pause" region is preserved.
///
/// # Arguments
///
/// * `voices` - Mutable slice of per-voice PCM buffers (same length, mono).
/// * `pauses` - Detected pause regions (from [`detect_pauses`]).
/// * `generator` - Breath noise generator (must have been created for the
///   same number of voices).
/// * `config` - Breath configuration.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid.
pub fn insert_breath_sounds(
    voices: &mut [Vec<f32>],
    pauses: &[PauseRegion],
    generator: &mut BreathGenerator,
    config: &BreathConfig,
) -> Result<(), KokoroError> {
    config.validate()?;

    if voices.is_empty() || pauses.is_empty() {
        return Ok(());
    }

    let sr = config.sample_rate;
    let breath_samples = (config.breath_duration_ms * 0.001 * sr).round() as usize;
    if breath_samples == 0 {
        return Ok(());
    }

    let max_stagger_samples = (config.stagger_ms * 0.001 * sr).round() as usize;

    for pause in pauses {
        // Only insert breath if the pause is long enough.
        if pause.duration < breath_samples / 2 {
            continue;
        }

        let actual_breath_len = breath_samples.min(pause.duration);

        for voice_idx in 0..voices.len().min(generator.n_voices) {
            // Per-voice stagger: deterministic offset derived from voice index
            // and pause start position.
            let stagger = if max_stagger_samples > 0 {
                let mut stagger_rng =
                    Lcg::new(voice_idx, 0x5748_8E47_0000_0000 | pause.start as u64);
                (stagger_rng.next_f32() * max_stagger_samples as f32).round() as usize
            } else {
                0
            };

            let insert_pos = pause.start.saturating_add(stagger);
            let voice_len = voices[voice_idx].len();

            if insert_pos >= voice_len {
                continue;
            }

            // Clamp breath length to fit within the voice buffer.
            let usable_len = actual_breath_len.min(voice_len - insert_pos);

            let breath = generator.generate(voice_idx, usable_len, config.breath_noise_level);

            // Mix (add) breath into the voice buffer.
            for (j, &b) in breath.iter().enumerate() {
                let idx = insert_pos + j;
                if idx < voice_len {
                    let mixed = voices[voice_idx][idx] + b;
                    voices[voice_idx][idx] = if mixed.is_finite() { mixed } else { 0.0 };
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BreathConfig tests ------------------------------------------------

    #[test]
    fn test_default_config_is_valid() {
        let config = BreathConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = BreathConfig::new()
            .with_noise_level(0.05)
            .with_duration_ms(120.0)
            .with_stagger_ms(50.0)
            .with_sample_rate(48000.0)
            .with_pause_threshold(0.005);
        assert!(config.validate().is_ok());
        assert!((config.breath_noise_level - 0.05).abs() < 1e-6);
        assert!((config.breath_duration_ms - 120.0).abs() < 1e-6);
        assert!((config.stagger_ms - 50.0).abs() < 1e-6);
        assert!((config.sample_rate - 48000.0).abs() < 1e-6);
        assert!((config.pause_threshold - 0.005).abs() < 1e-6);
    }

    #[test]
    fn test_config_validation_rejects_invalid() {
        // noise_level out of range
        assert!(BreathConfig::new()
            .with_noise_level(-0.01)
            .validate()
            .is_err());
        assert!(BreathConfig::new()
            .with_noise_level(0.3)
            .validate()
            .is_err());
        assert!(BreathConfig::new()
            .with_noise_level(f32::NAN)
            .validate()
            .is_err());

        // duration out of range
        assert!(BreathConfig::new()
            .with_duration_ms(5.0)
            .validate()
            .is_err());
        assert!(BreathConfig::new()
            .with_duration_ms(600.0)
            .validate()
            .is_err());
        assert!(BreathConfig::new()
            .with_duration_ms(f32::INFINITY)
            .validate()
            .is_err());

        // stagger out of range
        assert!(BreathConfig::new()
            .with_stagger_ms(-1.0)
            .validate()
            .is_err());
        assert!(BreathConfig::new()
            .with_stagger_ms(300.0)
            .validate()
            .is_err());

        // sample rate out of range
        assert!(BreathConfig::new()
            .with_sample_rate(100.0)
            .validate()
            .is_err());
        assert!(BreathConfig::new()
            .with_sample_rate(200_000.0)
            .validate()
            .is_err());
    }

    // -- Pause detection tests ---------------------------------------------

    #[test]
    fn test_detect_pauses_in_silence() {
        let config = BreathConfig::default();
        let audio = vec![0.0f32; 24000]; // 1 second of silence
        let pauses = detect_pauses(&audio, &config);
        assert!(!pauses.is_empty(), "should detect pause in silence");
        assert_eq!(pauses[0].start, 0);
    }

    #[test]
    fn test_detect_pauses_in_loud_signal() {
        let config = BreathConfig::default();
        // Constant amplitude well above threshold.
        let audio: Vec<f32> = (0..24000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let pauses = detect_pauses(&audio, &config);
        assert!(
            pauses.is_empty(),
            "should not detect pauses in loud signal, got {} pauses",
            pauses.len(),
        );
    }

    #[test]
    fn test_detect_pauses_finds_gap() {
        let config = BreathConfig::default();
        let sr = 24000usize;
        let mut audio = vec![0.0f32; sr]; // 1 second

        // Loud signal with a gap of silence in the middle.
        for i in 0..sr {
            if !(8000..16000).contains(&i) {
                audio[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin();
            }
            // Samples 8000..16000 stay at 0 (silence).
        }

        let pauses = detect_pauses(&audio, &config);
        assert!(!pauses.is_empty(), "should detect the silent gap");

        // The detected pause should overlap with the 8000..16000 region.
        let found = pauses
            .iter()
            .any(|p| p.start < 16000 && p.start + p.duration > 8000);
        assert!(found, "pause should overlap [8000, 16000]");
    }

    #[test]
    fn test_detect_pauses_empty_audio() {
        let config = BreathConfig::default();
        let pauses = detect_pauses(&[], &config);
        assert!(pauses.is_empty());
    }

    #[test]
    fn test_detect_pauses_handles_nan() {
        let config = BreathConfig::default();
        let mut audio = vec![0.5f32; 2400];
        audio[100] = f32::NAN;
        audio[200] = f32::INFINITY;
        // Should not panic; NaN/Inf treated as 0.
        let _pauses = detect_pauses(&audio, &config);
    }

    // -- BreathGenerator tests ---------------------------------------------

    #[test]
    fn test_generator_creates_ok() {
        let config = BreathConfig::default();
        let bgen = BreathGenerator::new(&config, 4);
        assert!(bgen.is_ok());
    }

    #[test]
    fn test_generator_zero_voices_errors() {
        let config = BreathConfig::default();
        assert!(BreathGenerator::new(&config, 0).is_err());
    }

    #[test]
    fn test_generator_output_length() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 2).unwrap();
        let breath = bgen.generate(0, 1920, 0.02);
        assert_eq!(breath.len(), 1920);
    }

    #[test]
    fn test_generator_output_is_finite() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 3).unwrap();
        for voice in 0..3 {
            let breath = bgen.generate(voice, 4800, 0.05);
            for (j, &s) in breath.iter().enumerate() {
                assert!(s.is_finite(), "voice {voice} sample {j} is non-finite: {s}");
            }
        }
    }

    #[test]
    fn test_generator_output_bounded() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 2).unwrap();
        let breath = bgen.generate(0, 4800, 0.02);
        let max_abs = breath.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        // Breath amplitude should be bounded by the noise level (with some
        // headroom from the filter).
        assert!(
            max_abs < 0.1,
            "breath max amplitude {max_abs} unexpectedly large",
        );
    }

    #[test]
    fn test_generator_envelope_shape() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 1).unwrap();
        let breath = bgen.generate(0, 2000, 1.0); // high level for easy measurement

        // First and last samples should be near zero (envelope fade).
        assert!(
            breath[0].abs() < 0.05,
            "first sample should be near zero, got {}",
            breath[0],
        );
        let last = breath[breath.len() - 1];
        assert!(
            last.abs() < 0.05,
            "last sample should be near zero, got {last}",
        );

        // Middle samples should have more energy.
        let mid = breath.len() / 2;
        let mid_rms: f32 = breath[mid - 50..mid + 50]
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            / 100.0;
        let edge_rms: f32 = breath[..100].iter().map(|s| s * s).sum::<f32>() / 100.0;
        assert!(
            mid_rms > edge_rms,
            "middle RMS ({mid_rms}) should exceed edge RMS ({edge_rms})",
        );
    }

    #[test]
    fn test_generator_voices_differ() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 3).unwrap();
        let b0 = bgen.generate(0, 2400, 0.02);
        bgen.reset();
        let mut bgen2 = BreathGenerator::new(&config, 3).unwrap();
        let b1 = bgen2.generate(1, 2400, 0.02);

        // Different voices should produce different noise sequences.
        let diff: f32 = b0
            .iter()
            .zip(b1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / b0.len() as f32;
        assert!(
            diff > 1e-6,
            "different voices should produce different breath noise, diff = {diff}",
        );
    }

    #[test]
    fn test_generator_invalid_voice_index() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 2).unwrap();
        let breath = bgen.generate(5, 1000, 0.02);
        assert!(breath.is_empty(), "out-of-range voice should return empty");
    }

    // -- Breath insertion tests --------------------------------------------

    #[test]
    fn test_insert_breath_sounds_basic() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 2).unwrap();

        // Two voices with silence (easy pause detection).
        let mut voices = vec![vec![0.0f32; 12000], vec![0.0f32; 12000]];
        let pauses = vec![PauseRegion {
            start: 2000,
            duration: 4000,
        }];

        let result = insert_breath_sounds(&mut voices, &pauses, &mut bgen, &config);
        assert!(result.is_ok());

        // The pause region should now have some non-zero samples from breath.
        let has_breath = voices[0][2000..6000].iter().any(|&s| s.abs() > 1e-8);
        assert!(has_breath, "breath should be inserted in pause region");
    }

    #[test]
    fn test_insert_breath_sounds_stagger() {
        let config = BreathConfig::new().with_stagger_ms(30.0);
        let mut bgen = BreathGenerator::new(&config, 3).unwrap();

        let mut voices = vec![
            vec![0.0f32; 24000],
            vec![0.0f32; 24000],
            vec![0.0f32; 24000],
        ];
        let pauses = vec![PauseRegion {
            start: 5000,
            duration: 8000,
        }];

        insert_breath_sounds(&mut voices, &pauses, &mut bgen, &config).unwrap();

        // Find the first non-zero sample in each voice's pause region.
        let first_nonzero = |voice: &[f32], start: usize, end: usize| -> Option<usize> {
            voice[start..end]
                .iter()
                .position(|&s| s.abs() > 1e-10)
                .map(|p| p + start)
        };

        let p0 = first_nonzero(&voices[0], 5000, 13000);
        let p1 = first_nonzero(&voices[1], 5000, 13000);
        let p2 = first_nonzero(&voices[2], 5000, 13000);

        // At least two voices should start at different positions (stagger).
        if let (Some(a), Some(b), Some(c)) = (p0, p1, p2) {
            let all_same = a == b && b == c;
            assert!(
                !all_same,
                "breath starts should be staggered: v0={a}, v1={b}, v2={c}",
            );
        }
    }

    #[test]
    fn test_insert_breath_sounds_empty_pauses() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 2).unwrap();
        let mut voices = vec![vec![0.5f32; 1000], vec![0.5f32; 1000]];
        let original = voices.clone();

        insert_breath_sounds(&mut voices, &[], &mut bgen, &config).unwrap();

        // No pauses -> audio unchanged.
        assert_eq!(voices, original);
    }

    #[test]
    fn test_insert_breath_sounds_all_finite() {
        let config = BreathConfig::default();
        let mut bgen = BreathGenerator::new(&config, 2).unwrap();

        let mut voices = vec![vec![0.0f32; 12000], vec![0.0f32; 12000]];
        let pauses = vec![
            PauseRegion {
                start: 1000,
                duration: 3000,
            },
            PauseRegion {
                start: 7000,
                duration: 2000,
            },
        ];

        insert_breath_sounds(&mut voices, &pauses, &mut bgen, &config).unwrap();

        for (vi, voice) in voices.iter().enumerate() {
            for (j, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "voice {vi} sample {j} is non-finite: {s}");
            }
        }
    }

    // -- End-to-end integration test ---------------------------------------

    #[test]
    fn test_end_to_end_detect_and_insert() {
        let config = BreathConfig::default();
        let sr = 24000usize;

        // Create audio with a clear silent gap.
        let mut audio = vec![0.0f32; sr];
        for i in 0..sr {
            if !(6000..18000).contains(&i) {
                audio[i] = 0.3 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr as f32).sin();
            }
        }

        // Detect pauses from a reference copy.
        let pauses = detect_pauses(&audio, &config);
        assert!(!pauses.is_empty(), "should detect the gap");

        // Create 3 voices from the same audio.
        let mut voices = vec![audio.clone(), audio.clone(), audio.clone()];
        let mut bgen = BreathGenerator::new(&config, 3).unwrap();

        insert_breath_sounds(&mut voices, &pauses, &mut bgen, &config).unwrap();

        // Verify all output is finite.
        for voice in &voices {
            for &s in voice {
                assert!(s.is_finite());
            }
        }

        // Verify that breath was added in the gap region.
        let gap_energy: f32 = voices[0][8000..16000].iter().map(|s| s * s).sum::<f32>();
        assert!(
            gap_energy > 1e-8,
            "gap should have breath energy, got {gap_energy}",
        );
    }
}
