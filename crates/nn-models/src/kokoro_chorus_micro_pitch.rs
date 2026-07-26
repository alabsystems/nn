// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Micro-pitch variation and drift for natural chorus shimmer.
//!
//! Real chorus singers have tiny, uncorrelated pitch wander that creates the
//! characteristic "shimmer" heard in live ensembles. This is distinct from:
//!
//! - **Detuning** (static pitch offset): fixed beating frequencies, constant.
//! - **Vibrato** (periodic LFO modulation): regular, predictable pitch wobble.
//! - **Micro-pitch drift** (this module): slow, random, 1/f-like pitch wandering.
//!
//! The micro-pitch drift sits "underneath" vibrato and detuning — it is the
//! background instability that makes a choir sound alive rather than synthetic.
//! Even with vibrato and detuning, a chorus without micro-pitch drift sounds
//! too stable and machine-like.
//!
//! # Algorithm
//!
//! 1. Generate slow 1/f noise via Voss-McCartney (sum of octave-band generators).
//! 2. Scale by `drift_cents` to get a pitch trajectory in cents.
//! 3. Apply partial correlation between voices (shared + independent components).
//! 4. Smooth the trajectory via exponential moving average to avoid clicks.
//! 5. Apply pitch shift via variable-rate delay line with linear interpolation.
//!
//! # References
//!
//! - Voss, R.F. & Clarke, J. "1/f noise in music and speech." Nature, 258, 1975.
//! - Boulanger, R. & Lazzarini, V. "The Audio Programming Book." MIT Press, 2011.
//! - Sundberg, J. "The Science of the Singing Voice." 1987. (pitch instability norms)

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for per-voice micro-pitch drift.
///
/// Controls the amplitude, rate, correlation, and smoothing of slow random
/// pitch wandering applied to each chorus voice.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MicroPitchConfig {
    /// Maximum pitch wander in cents (peak deviation from nominal pitch).
    ///
    /// Typical values: 2-8 cents for subtle shimmer, 10-20 cents for
    /// noticeable drift. Must be in [0.0, 50.0] and finite. Default: `5.0`.
    pub drift_cents: f32,

    /// Rate of pitch wandering in Hz (how fast the pitch drifts).
    ///
    /// Lower values give slower, more glacial drift; higher values give
    /// faster, more nervous wandering. Must be in [0.01, 5.0] and finite.
    /// Default: `0.3`.
    pub drift_rate_hz: f32,

    /// Per-voice seed offset. Each voice gets `base_seed + voice_index`
    /// as its random seed, producing different drift patterns.
    /// Default: `0`.
    pub per_voice_seed: u64,

    /// Cross-voice correlation factor.
    ///
    /// 0.0 = fully independent drift per voice (maximum shimmer).
    /// 1.0 = all voices drift together (no shimmer, just unison wander).
    /// Typical: 0.1-0.3 for natural choral sound.
    /// Must be in [0.0, 1.0] and finite. Default: `0.2`.
    pub correlation: f32,

    /// Smoothing time constant in milliseconds.
    ///
    /// Applies exponential moving average to the drift trajectory to
    /// prevent abrupt pitch jumps that would cause audible clicks.
    /// Must be in [1.0, 500.0] and finite. Default: `50.0`.
    pub smoothing_ms: f32,
}

impl Default for MicroPitchConfig {
    fn default() -> Self {
        Self {
            drift_cents: 5.0,
            drift_rate_hz: 0.3,
            per_voice_seed: 0,
            correlation: 0.2,
            smoothing_ms: 50.0,
        }
    }
}

impl MicroPitchConfig {
    /// Create a new micro-pitch configuration with validation.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is non-finite
    /// or outside its valid range.
    pub fn new(
        drift_cents: f32,
        drift_rate_hz: f32,
        per_voice_seed: u64,
        correlation: f32,
        smoothing_ms: f32,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            drift_cents,
            drift_rate_hz,
            per_voice_seed,
            correlation,
            smoothing_ms,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate this configuration.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.drift_cents.is_finite() || !(0.0..=50.0).contains(&self.drift_cents) {
            return Err(KokoroError::InvalidConfig {
                field: "drift_cents",
                reason: format!(
                    "must be finite and in [0.0, 50.0], got {}",
                    self.drift_cents,
                ),
            });
        }
        if !self.drift_rate_hz.is_finite() || !(0.01..=5.0).contains(&self.drift_rate_hz) {
            return Err(KokoroError::InvalidConfig {
                field: "drift_rate_hz",
                reason: format!(
                    "must be finite and in [0.01, 5.0], got {}",
                    self.drift_rate_hz,
                ),
            });
        }
        if !self.correlation.is_finite() || !(0.0..=1.0).contains(&self.correlation) {
            return Err(KokoroError::InvalidConfig {
                field: "correlation",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.correlation),
            });
        }
        if !self.smoothing_ms.is_finite() || !(1.0..=500.0).contains(&self.smoothing_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "smoothing_ms",
                reason: format!(
                    "must be finite and in [1.0, 500.0], got {}",
                    self.smoothing_ms,
                ),
            });
        }
        Ok(())
    }

    /// Subtle micro-pitch drift: barely perceptible, adds life.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            drift_cents: 2.0,
            drift_rate_hz: 0.2,
            per_voice_seed: 0,
            correlation: 0.2,
            smoothing_ms: 60.0,
        }
    }

    /// Chorus shimmer: noticeable drift that adds sparkle and width.
    #[must_use]
    pub fn chorus_shimmer() -> Self {
        Self {
            drift_cents: 8.0,
            drift_rate_hz: 0.5,
            per_voice_seed: 0,
            correlation: 0.15,
            smoothing_ms: 40.0,
        }
    }

    /// Slow, wide drift: glacial pitch wandering for ambient textures.
    #[must_use]
    pub fn drift() -> Self {
        Self {
            drift_cents: 15.0,
            drift_rate_hz: 0.1,
            per_voice_seed: 0,
            correlation: 0.1,
            smoothing_ms: 80.0,
        }
    }

    /// Tight micro-pitch: very small, fast variations for dense unison.
    #[must_use]
    pub fn tight() -> Self {
        Self {
            drift_cents: 1.0,
            drift_rate_hz: 1.0,
            per_voice_seed: 0,
            correlation: 0.3,
            smoothing_ms: 30.0,
        }
    }

    /// Reset the seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.per_voice_seed = seed;
        self
    }
}

// ---------------------------------------------------------------------------
// Voss-McCartney pink noise generator
// ---------------------------------------------------------------------------

/// Number of octave-band generators for Voss-McCartney pink noise.
/// 8 octaves covers the typical audible range of drift frequencies.
const PINK_OCTAVES: usize = 8;

/// Per-voice pink noise generator using the Voss-McCartney algorithm.
///
/// Produces 1/f (pink) noise by summing multiple octave-band white noise
/// generators that update at geometrically spaced intervals. The result
/// is a signal with equal energy per octave — the characteristic spectrum
/// of natural pitch instability in human voices.
struct PinkNoiseGen {
    /// Per-octave random state (simple xorshift).
    octave_values: [f32; PINK_OCTAVES],
    /// Per-octave RNG state.
    rng_states: [u64; PINK_OCTAVES],
    /// Sample counter for octave update scheduling.
    counter: u64,
}

impl PinkNoiseGen {
    /// Create a new pink noise generator with the given seed.
    fn new(seed: u64) -> Self {
        let mut rng_states = [0u64; PINK_OCTAVES];
        let mut octave_values = [0.0f32; PINK_OCTAVES];
        for (i, state) in rng_states.iter_mut().enumerate() {
            // Derive per-octave seed to avoid correlation.
            *state = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64 + 1);
            if *state == 0 {
                *state = 1;
            }
        }
        // Initialize octave values.
        for i in 0..PINK_OCTAVES {
            octave_values[i] = xorshift_f32(&mut rng_states[i]);
        }
        Self {
            octave_values,
            rng_states,
            counter: 0,
        }
    }

    /// Generate the next pink noise sample in [-1.0, 1.0].
    fn next_sample(&mut self) -> f32 {
        // Voss-McCartney: octave k updates every 2^k samples.
        // At each step, update the octave corresponding to the
        // lowest set bit that changed.
        let prev = self.counter;
        self.counter = self.counter.wrapping_add(1);
        let changed = prev ^ self.counter;

        for k in 0..PINK_OCTAVES {
            if changed & (1u64 << k) != 0 {
                self.octave_values[k] = xorshift_f32(&mut self.rng_states[k]);
            }
        }

        // Sum and normalize to [-1, 1].
        let sum: f32 = self.octave_values.iter().sum();
        let normalized = sum / PINK_OCTAVES as f32;
        // Clamp for safety (should already be in range).
        normalized.clamp(-1.0, 1.0)
    }
}

/// Simple xorshift64 PRNG returning a value in [-1.0, 1.0].
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    // Map to [-1.0, 1.0] using the lower 24 bits.
    let bits = (s & 0x00FF_FFFF) as f32 / (0x00FF_FFFF as f32);
    bits * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Per-voice micro-pitch drift processor.
///
/// Maintains per-voice state (noise generator, smoothed pitch offset, delay
/// line buffer) across calls to `process_voices`. Create one instance and
/// reuse it across chunks for streaming continuity.
pub struct MicroPitchProcessor {
    config: MicroPitchConfig,
    /// Per-voice pink noise generators (independent component).
    voice_noise: Vec<PinkNoiseGen>,
    /// Shared pink noise generator (correlated component).
    shared_noise: PinkNoiseGen,
    /// Per-voice smoothed pitch offset in cents.
    smoothed_offsets: Vec<f32>,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Decimation counter: we only need to update the noise at drift_rate_hz,
    /// not at the full sample rate. This is the step size in samples.
    noise_step: usize,
    /// Per-voice sub-sample counter for noise decimation.
    voice_subcounters: Vec<usize>,
    /// Per-voice raw (pre-smoothing) offset in cents.
    raw_offsets: Vec<f32>,
}

impl MicroPitchProcessor {
    /// Create a new micro-pitch processor.
    ///
    /// # Arguments
    ///
    /// * `config` - Micro-pitch configuration.
    /// * `n_voices` - Number of chorus voices.
    /// * `sample_rate` - Audio sample rate in Hz.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails.
    pub fn new(
        config: &MicroPitchConfig,
        n_voices: usize,
        sample_rate: u32,
    ) -> Result<Self, KokoroError> {
        config.validate()?;
        if sample_rate == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: "must be > 0".to_string(),
            });
        }

        let mut voice_noise = Vec::with_capacity(n_voices);
        for i in 0..n_voices {
            let seed = config
                .per_voice_seed
                .wrapping_add(i as u64)
                .wrapping_mul(2654435761);
            voice_noise.push(PinkNoiseGen::new(seed));
        }
        let shared_seed = config
            .per_voice_seed
            .wrapping_mul(1442695040888963407)
            .wrapping_add(7);
        let shared_noise = PinkNoiseGen::new(shared_seed);

        // Noise update rate: we update the noise trajectory at a rate
        // proportional to drift_rate_hz. Higher drift_rate = more updates/sec.
        // Minimum step of 1 to avoid division by zero.
        let updates_per_sec = (config.drift_rate_hz * 20.0).max(1.0);
        let noise_step = ((sample_rate as f32 / updates_per_sec) as usize).max(1);

        Ok(Self {
            config: config.clone(),
            voice_noise,
            shared_noise,
            smoothed_offsets: vec![0.0; n_voices],
            sample_rate,
            noise_step,
            voice_subcounters: vec![0; n_voices],
            raw_offsets: vec![0.0; n_voices],
        })
    }

    /// Process all voice buffers, applying micro-pitch drift in-place.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidInput` if voice count mismatches.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) -> Result<(), KokoroError> {
        let n_voices = self.voice_noise.len();
        if voices.len() != n_voices {
            return Err(KokoroError::InvalidInput(format!(
                "voices length {} != configured voice count {}",
                voices.len(),
                n_voices,
            )));
        }

        if voices.is_empty() || self.config.drift_cents < 1e-4 {
            return Ok(());
        }

        let sr = f64::from(self.sample_rate);
        let smoothing_alpha = smoothing_coefficient(self.config.smoothing_ms, self.sample_rate);
        let corr = f64::from(self.config.correlation.clamp(0.0, 1.0));
        let indep = 1.0 - corr;

        for voice_idx in 0..n_voices {
            let len = voices[voice_idx].len();
            if len == 0 {
                continue;
            }

            let mut resampled = Vec::with_capacity(len);
            let mut src_pos: f64 = 0.0;

            for i in 0..len {
                // Update noise at decimated rate.
                self.voice_subcounters[voice_idx] += 1;
                if self.voice_subcounters[voice_idx] >= self.noise_step {
                    self.voice_subcounters[voice_idx] = 0;

                    let indep_noise = f64::from(self.voice_noise[voice_idx].next_sample());
                    let shared_noise = f64::from(self.shared_noise.next_sample());
                    let combined = indep * indep_noise + corr * shared_noise;

                    self.raw_offsets[voice_idx] =
                        (combined * f64::from(self.config.drift_cents)) as f32;
                }

                // Exponential smoothing of the pitch offset.
                let target = self.raw_offsets[voice_idx];
                let prev = self.smoothed_offsets[voice_idx];
                let smoothed = prev + smoothing_alpha * (target - prev);
                self.smoothed_offsets[voice_idx] =
                    if smoothed.is_finite() { smoothed } else { 0.0 };

                // Convert cents offset to instantaneous resampling rate.
                let cents = f64::from(self.smoothed_offsets[voice_idx]);
                let rate = (2.0f64).powf(cents / 1200.0);

                // Variable-rate delay line: advance source position by `rate`.
                let src_idx = src_pos.floor() as isize;
                let frac = (src_pos - src_idx as f64) as f32;

                let pcm = &voices[voice_idx];

                // Linear interpolation with boundary clamping.
                let s0 = sample_clamped(pcm, src_idx);
                let s1 = sample_clamped(pcm, src_idx + 1);
                let sample = s0 + frac * (s1 - s0);
                resampled.push(if sample.is_finite() { sample } else { 0.0 });

                src_pos += rate;

                // Prevent source position from running too far ahead or behind.
                // If drift accumulates beyond buffer length, clamp.
                let max_lead = len as f64 + (sr * 0.1);
                if src_pos > max_lead {
                    src_pos = i as f64 + 1.0;
                }
                if src_pos < -(sr * 0.1) {
                    src_pos = i as f64;
                }
            }

            // Replace the voice buffer with the pitch-shifted version.
            // Trim or pad to match original length.
            voices[voice_idx].clear();
            voices[voice_idx].extend_from_slice(&resampled[..resampled.len().min(len)]);
            while voices[voice_idx].len() < len {
                voices[voice_idx].push(0.0);
            }
        }

        Ok(())
    }

    /// Get the current smoothed pitch offset per voice in cents.
    #[must_use]
    pub fn get_current_offsets(&self) -> Vec<f32> {
        self.smoothed_offsets.clone()
    }

    /// Reset all per-voice state (noise generators, smoothed offsets).
    pub fn reset(&mut self) {
        for i in 0..self.voice_noise.len() {
            let seed = self
                .config
                .per_voice_seed
                .wrapping_add(i as u64)
                .wrapping_mul(2654435761);
            self.voice_noise[i] = PinkNoiseGen::new(seed);
            self.smoothed_offsets[i] = 0.0;
            self.voice_subcounters[i] = 0;
            self.raw_offsets[i] = 0.0;
        }
        let shared_seed = self
            .config
            .per_voice_seed
            .wrapping_mul(1442695040888963407)
            .wrapping_add(7);
        self.shared_noise = PinkNoiseGen::new(shared_seed);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute exponential moving average coefficient from time constant in ms.
///
/// `alpha = 1 - exp(-1 / (tau_ms / 1000 * sample_rate))`
fn smoothing_coefficient(tau_ms: f32, sample_rate: u32) -> f32 {
    let tau_samples = (tau_ms / 1000.0) * sample_rate as f32;
    if !tau_samples.is_finite() || tau_samples < 1.0 {
        return 1.0; // No smoothing.
    }
    let alpha = 1.0 - (-1.0 / tau_samples).exp();
    if alpha.is_finite() {
        alpha
    } else {
        1.0
    }
}

/// Read a sample from the buffer with boundary clamping.
#[inline]
fn sample_clamped(buf: &[f32], idx: isize) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    if idx < 0 {
        buf[0]
    } else if (idx as usize) >= buf.len() {
        buf[buf.len() - 1]
    } else {
        buf[idx as usize]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = MicroPitchConfig::default();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_presets_valid() {
        MicroPitchConfig::subtle().validate().expect("subtle");
        MicroPitchConfig::chorus_shimmer()
            .validate()
            .expect("chorus_shimmer");
        MicroPitchConfig::drift().validate().expect("drift");
        MicroPitchConfig::tight().validate().expect("tight");
    }

    #[test]
    fn test_config_validation_rejects_invalid() {
        assert!(MicroPitchConfig::new(-1.0, 0.3, 0, 0.2, 50.0).is_err());
        assert!(MicroPitchConfig::new(51.0, 0.3, 0, 0.2, 50.0).is_err());
        assert!(MicroPitchConfig::new(f32::NAN, 0.3, 0, 0.2, 50.0).is_err());
        assert!(MicroPitchConfig::new(5.0, 0.005, 0, 0.2, 50.0).is_err());
        assert!(MicroPitchConfig::new(5.0, 6.0, 0, 0.2, 50.0).is_err());
        assert!(MicroPitchConfig::new(5.0, 0.3, 0, -0.1, 50.0).is_err());
        assert!(MicroPitchConfig::new(5.0, 0.3, 0, 1.1, 50.0).is_err());
        assert!(MicroPitchConfig::new(5.0, 0.3, 0, 0.2, 0.5).is_err());
        assert!(MicroPitchConfig::new(5.0, 0.3, 0, 0.2, 501.0).is_err());
        assert!(MicroPitchConfig::new(5.0, f32::INFINITY, 0, 0.2, 50.0).is_err());
    }

    #[test]
    fn test_config_validation_accepts_valid() {
        assert!(MicroPitchConfig::new(0.0, 0.01, 0, 0.0, 1.0).is_ok());
        assert!(MicroPitchConfig::new(50.0, 5.0, 99, 1.0, 500.0).is_ok());
        assert!(MicroPitchConfig::new(5.0, 0.3, 42, 0.2, 50.0).is_ok());
    }

    #[test]
    fn test_zero_drift_is_near_identity() {
        let config = MicroPitchConfig::new(0.0, 0.3, 0, 0.2, 50.0).expect("valid config");
        let mut proc = MicroPitchProcessor::new(&config, 3, 24000).expect("valid proc");
        let original: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut voices = vec![original.clone(), original.clone(), original.clone()];
        proc.process_voices(&mut voices).expect("process ok");

        // With 0 drift_cents, output should equal input.
        for (vi, voice) in voices.iter().enumerate() {
            for (j, (&got, &expected)) in voice.iter().zip(original.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-5,
                    "voice {vi} sample {j}: got {got}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn test_drift_changes_audio() {
        let config = MicroPitchConfig::chorus_shimmer();
        let mut proc = MicroPitchProcessor::new(&config, 3, 24000).expect("valid proc");
        let signal: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal.clone(), signal.clone(), signal.clone()];
        proc.process_voices(&mut voices).expect("process ok");

        // At least one voice should differ from the original.
        let diff: f32 = voices[1]
            .iter()
            .zip(signal.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / voices[1].len() as f32;

        assert!(
            diff > 1e-6,
            "micro-pitch drift should modify audio, mean diff = {diff}",
        );
    }

    #[test]
    fn test_preserves_buffer_length() {
        let config = MicroPitchConfig::default();
        let mut proc = MicroPitchProcessor::new(&config, 4, 24000).expect("valid proc");
        let len = 6000;
        let signal: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut voices = vec![signal; 4];
        proc.process_voices(&mut voices).expect("process ok");

        for (i, voice) in voices.iter().enumerate() {
            assert_eq!(voice.len(), len, "voice {i} length should be preserved");
        }
    }

    #[test]
    fn test_voices_differ_from_each_other() {
        let config = MicroPitchConfig::new(10.0, 0.5, 0, 0.0, 30.0).expect("valid config");
        let mut proc = MicroPitchProcessor::new(&config, 4, 24000).expect("valid proc");
        let signal: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal; 4];
        proc.process_voices(&mut voices).expect("process ok");

        // Voices 1 and 2 should differ (different noise seeds, zero correlation).
        let diff: f32 = voices[1]
            .iter()
            .zip(voices[2].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / voices[1].len() as f32;

        assert!(
            diff > 1e-6,
            "different voices should drift differently, mean diff = {diff}",
        );
    }

    #[test]
    fn test_full_correlation_voices_drift_together() {
        let config = MicroPitchConfig::new(10.0, 0.5, 0, 1.0, 30.0).expect("valid config");
        let mut proc = MicroPitchProcessor::new(&config, 3, 24000).expect("valid proc");
        let signal: Vec<f32> = (0..6000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal; 3];
        proc.process_voices(&mut voices).expect("process ok");

        // With correlation=1.0, voices should be very similar to each other
        // (though not identical due to different initial noise seeds).
        let diff_12: f32 = voices[0]
            .iter()
            .zip(voices[1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / voices[0].len() as f32;

        // This diff should be small relative to a zero-correlation scenario.
        // We just check it's below a reasonable threshold.
        assert!(
            diff_12 < 0.1,
            "fully correlated voices should drift similarly, mean diff = {diff_12}",
        );
    }

    #[test]
    fn test_no_nan_in_output() {
        let config = MicroPitchConfig::new(15.0, 1.0, 42, 0.3, 20.0).expect("valid config");
        let mut proc = MicroPitchProcessor::new(&config, 3, 24000).expect("valid proc");
        let signal: Vec<f32> = (0..12000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut voices = vec![signal; 3];
        proc.process_voices(&mut voices).expect("process ok");

        for (vi, voice) in voices.iter().enumerate() {
            for (j, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "voice {vi} sample {j} is non-finite: {s}");
            }
        }
    }

    #[test]
    fn test_get_current_offsets() {
        let config = MicroPitchConfig::default();
        let proc = MicroPitchProcessor::new(&config, 4, 24000).expect("valid proc");
        let offsets = proc.get_current_offsets();
        assert_eq!(offsets.len(), 4);
        // Initially all offsets are 0.
        for &o in &offsets {
            assert!((o).abs() < 1e-6, "initial offset should be 0, got {o}");
        }
    }

    #[test]
    fn test_reset_returns_to_initial_state() {
        let config = MicroPitchConfig::chorus_shimmer();
        let mut proc = MicroPitchProcessor::new(&config, 3, 24000).expect("valid proc");
        let signal: Vec<f32> = (0..6000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut voices = vec![signal; 3];

        // Process once to advance state.
        proc.process_voices(&mut voices).expect("process ok");
        let offsets_after = proc.get_current_offsets();
        let has_nonzero = offsets_after.iter().any(|&o| o.abs() > 1e-8);
        assert!(has_nonzero, "offsets should be nonzero after processing");

        // Reset.
        proc.reset();
        let offsets_reset = proc.get_current_offsets();
        for &o in &offsets_reset {
            assert!((o).abs() < 1e-6, "offset should be 0 after reset, got {o}");
        }
    }

    #[test]
    fn test_voice_count_mismatch_error() {
        let config = MicroPitchConfig::default();
        let mut proc = MicroPitchProcessor::new(&config, 3, 24000).expect("valid proc");
        let mut voices = vec![vec![0.0; 100], vec![0.0; 100]]; // 2 != 3
        assert!(proc.process_voices(&mut voices).is_err());
    }

    #[test]
    fn test_empty_voices_ok() {
        let config = MicroPitchConfig::default();
        let mut proc = MicroPitchProcessor::new(&config, 0, 24000).expect("valid proc");
        let mut voices: Vec<Vec<f32>> = vec![];
        proc.process_voices(&mut voices).expect("empty ok");
    }

    #[test]
    fn test_pink_noise_generator_bounded() {
        let mut noise_gen = PinkNoiseGen::new(12345);
        for _ in 0..10_000 {
            let sample = noise_gen.next_sample();
            assert!(
                (-1.0..=1.0).contains(&sample),
                "pink noise sample out of range: {sample}",
            );
        }
    }

    #[test]
    fn test_smoothing_coefficient_reasonable() {
        // Fast smoothing (small tau) -> alpha close to 1.
        let alpha_fast = smoothing_coefficient(1.0, 24000);
        assert!(
            alpha_fast > 0.01,
            "fast alpha should be significant: {alpha_fast}"
        );

        // Slow smoothing (large tau) -> alpha close to 0.
        let alpha_slow = smoothing_coefficient(500.0, 24000);
        assert!(alpha_slow < alpha_fast, "slow alpha should be < fast alpha");
        assert!(alpha_slow > 0.0, "slow alpha should be positive");
    }

    #[test]
    fn test_with_seed_builder() {
        let config = MicroPitchConfig::subtle().with_seed(42);
        assert_eq!(config.per_voice_seed, 42);
        config.validate().expect("seeded config should be valid");
    }

    #[test]
    fn test_preset_parameter_ranges() {
        let subtle = MicroPitchConfig::subtle();
        assert!(
            subtle.drift_cents <= 3.0,
            "subtle drift should be <= 3 cents"
        );

        let shimmer = MicroPitchConfig::chorus_shimmer();
        assert!(
            shimmer.drift_cents >= 5.0,
            "shimmer drift should be >= 5 cents"
        );

        let drift = MicroPitchConfig::drift();
        assert!(drift.drift_rate_hz <= 0.2, "drift rate should be slow");
        assert!(drift.drift_cents >= 10.0, "drift should be wide");

        let tight = MicroPitchConfig::tight();
        assert!(tight.drift_cents <= 2.0, "tight drift should be small");
        assert!(tight.drift_rate_hz >= 0.8, "tight rate should be fast");
    }
}
