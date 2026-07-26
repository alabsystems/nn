// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Allpass diffusion de-correlation for perceptually distinct chorus voices.
//!
//! When multiple chorus voices share the same source waveform (or nearly
//! identical waveforms), they can fuse perceptually into a single source
//! with comb-filtering artifacts. This module randomizes the **phase
//! response** of each voice independently via cascaded allpass diffusion
//! filters, creating the perception of separate sound sources without
//! altering pitch, timing, or spectral magnitude.
//!
//! # How it works
//!
//! Each voice passes through a cascade of `n_stages` first-order allpass
//! filters. Each allpass stage has a unique delay line and coefficient
//! derived from a deterministic per-voice seed. Because allpass filters
//! have unity magnitude at all frequencies, the spectrum of each voice is
//! preserved — only the phase is scrambled. Different voices receive
//! different delay/coefficient parameters, so their phase responses diverge,
//! breaking the coherence that causes perceptual fusion.
//!
//! # Allpass filter equation
//!
//! ```text
//! y[n] = -g * x[n] + x[n - d] + g * y[n - d]
//! ```
//!
//! where `g` is the diffusion coefficient and `d` is the delay in samples.
//! This is energy-preserving: `|H(e^jw)| = 1` for all w.
//!
//! # Frequency-dependent diffusion
//!
//! When enabled, the signal is split into two bands at 2 kHz. High
//! frequencies receive full diffusion (phase scrambling is most audible
//! there), while low frequencies receive reduced diffusion to preserve
//! bass coherence and punch.
//!
//! # References
//!
//! - Jot, J.-M. & Chaigne, A., "Digital Delay Networks for Designing
//!   Artificial Reverberators," AES 90th Convention, 1991.
//! - Gerzon, M., "Unitary (Energy-Preserving) Multichannel Networks with
//!   Feedback," Electronics Letters, 12(11), 1976.
//! - Dattorro, J., "Effect Design Part 1: Reverberator and Other Filters,"
//!   J. Audio Eng. Soc., 45(9), 1997.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64)
// ---------------------------------------------------------------------------

/// Splitmix64 PRNG — fast, deterministic, excellent avalanche properties.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64, voice_index: usize, stage_index: usize) -> Self {
        // Mix voice and stage indices into the seed for full decorrelation.
        let state = seed
            .wrapping_add(voice_index as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(stage_index as u64)
            .wrapping_mul(0x517C_C1B7_2722_0A95);
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a value in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ---------------------------------------------------------------------------
// Biquad crossover (Linkwitz-Riley-style lowpass/highpass at 2 kHz)
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

    fn highpass(cutoff_hz: f32, sample_rate: f32) -> Self {
        let omega = std::f32::consts::TAU * (cutoff_hz / sample_rate);
        let (sin_w, cos_w) = (omega.sin(), omega.cos());
        let alpha = sin_w / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0_inv = 1.0 / (1.0 + alpha);
        Self {
            b0: f32::midpoint(1.0, cos_w) * a0_inv,
            b1: (-(1.0 + cos_w)) * a0_inv,
            b2: f32::midpoint(1.0, cos_w) * a0_inv,
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
// Allpass diffusion stage
// ---------------------------------------------------------------------------

/// A single first-order allpass diffusion stage with an integer-sample delay.
///
/// Implements `y[n] = -g * x[n] + x[n-d] + g * y[n-d]` where `g` is the
/// diffusion coefficient and `d` is the delay in samples. This is
/// energy-preserving: the output has the same power as the input.
#[derive(Debug, Clone)]
struct AllpassDiffusionStage {
    /// Diffusion coefficient g in (-1, 1).
    coeff: f32,
    /// Circular delay buffer for input samples.
    x_buf: Vec<f32>,
    /// Circular delay buffer for output samples.
    y_buf: Vec<f32>,
    /// Write position in the circular buffer.
    write_pos: usize,
    /// Delay in samples (buffer length).
    delay: usize,
}

impl AllpassDiffusionStage {
    /// Create a new allpass stage.
    ///
    /// `delay_samples` must be >= 1. `coeff` is clamped to (-0.99, 0.99)
    /// for stability.
    fn new(delay_samples: usize, coeff: f32) -> Self {
        let delay = delay_samples.max(1);
        let coeff = coeff.clamp(-0.99, 0.99);
        let coeff = if !coeff.is_finite() { 0.0 } else { coeff };
        Self {
            coeff,
            x_buf: vec![0.0; delay],
            y_buf: vec![0.0; delay],
            write_pos: 0,
            delay,
        }
    }

    /// Process one sample through the allpass stage.
    ///
    /// `y[n] = -g * x[n] + x[n-d] + g * y[n-d]`
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        if !input.is_finite() {
            return 0.0;
        }

        // Read the delayed input and output.
        let read_pos = self.write_pos;
        let x_delayed = self.x_buf[read_pos];
        let y_delayed = self.y_buf[read_pos];

        // Allpass equation: y = -g*x + x[n-d] + g*y[n-d]
        let output = -self.coeff * input + x_delayed + self.coeff * y_delayed;

        // Guard against numerical blowup.
        let output = if !output.is_finite() { 0.0 } else { output };

        // Write current input and output into the delay buffers.
        self.x_buf[self.write_pos] = input;
        self.y_buf[self.write_pos] = output;

        // Advance write position.
        self.write_pos += 1;
        if self.write_pos >= self.delay {
            self.write_pos = 0;
        }

        output
    }

    fn reset(&mut self) {
        self.x_buf.fill(0.0);
        self.y_buf.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for allpass diffusion de-correlation.
///
/// Use builder methods or preset constructors (`subtle()`, `wide()`,
/// `maximum()`, `bass_safe()`). `#[non_exhaustive]` allows adding fields
/// without breaking downstream consumers.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecorrelationConfig {
    /// Number of cascaded allpass diffusion stages per voice.
    ///
    /// More stages = more thorough phase randomization.
    /// Range: [1, 32]. Default: 8.
    pub n_stages: usize,

    /// Maximum allpass delay in milliseconds.
    ///
    /// Controls the maximum delay line length. Longer delays create more
    /// audible de-correlation but may introduce perceptible pre-echo.
    /// Range: [0.5, 50.0]. Default: 5.0.
    pub max_delay_ms: f32,

    /// Diffusion amount in [0.0, 1.0].
    ///
    /// Controls the allpass coefficient magnitude. 0.0 = no diffusion
    /// (pass-through), 1.0 = maximum diffusion (heavy phase scrambling).
    /// Default: 0.7.
    pub diffusion: f32,

    /// Whether to apply frequency-dependent diffusion.
    ///
    /// When true, splits the signal at 2 kHz: high frequencies get full
    /// diffusion, low frequencies get reduced diffusion (0.3x) to preserve
    /// bass coherence. Default: true.
    pub frequency_dependent: bool,

    /// PRNG seed for deterministic per-voice delay values.
    ///
    /// Different seeds produce different delay/coefficient patterns.
    /// Same seed + same voice count = identical results across runs.
    /// Default: 42.
    pub per_voice_seed: u64,

    /// Dry/wet mix ratio.
    ///
    /// 0.0 = fully dry (bypass), 1.0 = fully wet (allpass only).
    /// Range: [0.0, 1.0]. Default: 0.5.
    pub mix: f32,
}

impl Default for DecorrelationConfig {
    fn default() -> Self {
        Self {
            n_stages: 8,
            max_delay_ms: 5.0,
            diffusion: 0.7,
            frequency_dependent: true,
            per_voice_seed: 42,
            mix: 0.5,
        }
    }
}

impl DecorrelationConfig {
    /// Create a new configuration with validation.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range
    /// or non-finite.
    pub fn new(
        n_stages: usize,
        max_delay_ms: f32,
        diffusion: f32,
        frequency_dependent: bool,
        per_voice_seed: u64,
        mix: f32,
    ) -> Result<Self, KokoroError> {
        let cfg = Self {
            n_stages,
            max_delay_ms,
            diffusion,
            frequency_dependent,
            per_voice_seed,
            mix,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !(1..=32).contains(&self.n_stages) {
            return Err(KokoroError::InvalidConfig {
                field: "n_stages",
                reason: format!("must be in [1, 32], got {}", self.n_stages),
            });
        }
        if !self.max_delay_ms.is_finite() || !(0.5..=50.0).contains(&self.max_delay_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "max_delay_ms",
                reason: format!(
                    "must be finite and in [0.5, 50.0], got {}",
                    self.max_delay_ms
                ),
            });
        }
        if !self.diffusion.is_finite() || !(0.0..=1.0).contains(&self.diffusion) {
            return Err(KokoroError::InvalidConfig {
                field: "diffusion",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.diffusion),
            });
        }
        if !self.mix.is_finite() || !(0.0..=1.0).contains(&self.mix) {
            return Err(KokoroError::InvalidConfig {
                field: "mix",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.mix),
            });
        }
        Ok(())
    }

    // -- Builder methods -------------------------------------------------------

    /// Set the number of allpass stages.
    #[must_use]
    pub fn with_n_stages(mut self, n: usize) -> Self {
        self.n_stages = n;
        self
    }

    /// Set the maximum delay in milliseconds.
    #[must_use]
    pub fn with_max_delay_ms(mut self, ms: f32) -> Self {
        self.max_delay_ms = ms;
        self
    }

    /// Set the diffusion amount.
    #[must_use]
    pub fn with_diffusion(mut self, d: f32) -> Self {
        self.diffusion = d;
        self
    }

    /// Enable or disable frequency-dependent diffusion.
    #[must_use]
    pub fn with_frequency_dependent(mut self, fd: bool) -> Self {
        self.frequency_dependent = fd;
        self
    }

    /// Set the per-voice PRNG seed.
    #[must_use]
    pub fn with_per_voice_seed(mut self, seed: u64) -> Self {
        self.per_voice_seed = seed;
        self
    }

    /// Set the dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, m: f32) -> Self {
        self.mix = m;
        self
    }

    // -- Presets ---------------------------------------------------------------

    /// Subtle de-correlation: light phase variation, preserves source character.
    ///
    /// 4 stages, 2ms max delay, 0.4 diffusion, 0.3 mix.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            n_stages: 4,
            max_delay_ms: 2.0,
            diffusion: 0.4,
            frequency_dependent: true,
            per_voice_seed: 42,
            mix: 0.3,
        }
    }

    /// Wide de-correlation: noticeable separation between voices.
    ///
    /// 8 stages, 7ms max delay, 0.75 diffusion, 0.6 mix.
    #[must_use]
    pub fn wide() -> Self {
        Self {
            n_stages: 8,
            max_delay_ms: 7.0,
            diffusion: 0.75,
            frequency_dependent: true,
            per_voice_seed: 42,
            mix: 0.6,
        }
    }

    /// Maximum de-correlation: aggressive phase scrambling for maximum
    /// perceived source separation.
    ///
    /// 16 stages, 12ms max delay, 0.95 diffusion, 0.8 mix.
    #[must_use]
    pub fn maximum() -> Self {
        Self {
            n_stages: 16,
            max_delay_ms: 12.0,
            diffusion: 0.95,
            frequency_dependent: true,
            per_voice_seed: 42,
            mix: 0.8,
        }
    }

    /// Bass-safe de-correlation: strong high-frequency decorrelation with
    /// bass coherence fully preserved.
    ///
    /// 8 stages, 5ms max delay, 0.8 diffusion, frequency-dependent on,
    /// 0.5 mix. Best for music where low-end phase coherence is critical.
    #[must_use]
    pub fn bass_safe() -> Self {
        Self {
            n_stages: 8,
            max_delay_ms: 5.0,
            diffusion: 0.8,
            frequency_dependent: true,
            per_voice_seed: 42,
            mix: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-voice allpass chain
// ---------------------------------------------------------------------------

/// A cascade of allpass diffusion stages for one voice.
#[derive(Debug, Clone)]
struct VoiceAllpassChain {
    stages: Vec<AllpassDiffusionStage>,
}

impl VoiceAllpassChain {
    /// Build a chain of `n_stages` allpass filters with deterministic
    /// delays and coefficients derived from the given seed, voice index,
    /// and sample rate.
    fn new(
        n_stages: usize,
        max_delay_ms: f32,
        diffusion: f32,
        seed: u64,
        voice_index: usize,
        sample_rate: f32,
    ) -> Self {
        let max_delay_samples = (max_delay_ms * 0.001 * sample_rate).round() as usize;
        let max_delay_samples = max_delay_samples.max(1);

        let stages = (0..n_stages)
            .map(|stage_idx| {
                let mut rng = SplitMix64::new(seed, voice_index, stage_idx);

                // Delay: between 1 and max_delay_samples, deterministic per
                // voice/stage.
                let range = max_delay_samples.saturating_sub(1).max(1);
                let delay = 1 + (rng.next_f32() * range as f32) as usize;
                let delay = delay.clamp(1, max_delay_samples);

                // Coefficient: sign alternates per stage for better diffusion,
                // magnitude is the configured diffusion amount with slight
                // per-stage variation.
                let sign = if stage_idx % 2 == 0 { 1.0 } else { -1.0 };
                let variation = 0.85 + 0.15 * rng.next_f32();
                let coeff = sign * diffusion * variation;

                AllpassDiffusionStage::new(delay, coeff)
            })
            .collect();

        Self { stages }
    }

    /// Process one sample through the full cascade.
    #[inline]
    fn process(&mut self, mut sample: f32) -> f32 {
        for stage in &mut self.stages {
            sample = stage.process(sample);
        }
        sample
    }

    fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Allpass diffusion de-correlation processor for multi-voice chorus.
///
/// Applies per-voice phase randomization via cascaded allpass networks.
/// Each voice gets a unique allpass chain so that identical input signals
/// emerge with different phase responses, creating the perception of
/// separate sound sources.
///
/// # Example
///
/// ```rust
/// use nn_models::kokoro_chorus_decorrelation::{DecorrelationConfig, DecorrelationProcessor};
///
/// let config = DecorrelationConfig::subtle();
/// let mut proc = DecorrelationProcessor::new(&config, 4, 24000.0).unwrap();
///
/// let mut voices: Vec<Vec<f32>> = vec![vec![0.5; 1024]; 4];
/// proc.process_voices(&mut voices);
/// ```
pub struct DecorrelationProcessor {
    config: DecorrelationConfig,
    n_voices: usize,
    sample_rate: f32,

    /// Per-voice allpass chains. One per voice.
    chains: Vec<VoiceAllpassChain>,

    /// Optional per-voice crossover filters for frequency-dependent mode.
    /// Each entry is (lowpass, highpass) for the 2 kHz split.
    crossovers: Option<Vec<(BiquadFilter, BiquadFilter)>>,

    /// High-frequency allpass chains (used only in frequency-dependent mode).
    /// Low frequencies get a scaled-down version of the same chain.
    hi_chains: Vec<VoiceAllpassChain>,
}

impl DecorrelationProcessor {
    /// Crossover frequency for frequency-dependent diffusion.
    const CROSSOVER_HZ: f32 = 2000.0;

    /// Low-frequency diffusion scaling factor (relative to configured diffusion).
    const LO_DIFFUSION_SCALE: f32 = 0.3;

    /// Create a new de-correlation processor.
    ///
    /// # Arguments
    ///
    /// * `config` — De-correlation parameters.
    /// * `n_voices` — Number of chorus voices to process.
    /// * `sample_rate` — Audio sample rate in Hz.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails or
    /// `KokoroError::InvalidInput` if `n_voices` is zero or `sample_rate`
    /// is non-positive/non-finite.
    pub fn new(
        config: &DecorrelationConfig,
        n_voices: usize,
        sample_rate: f32,
    ) -> Result<Self, KokoroError> {
        config.validate()?;

        if n_voices == 0 {
            return Err(KokoroError::InvalidInput("n_voices must be >= 1".into()));
        }
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidInput(format!(
                "sample_rate must be finite and positive, got {sample_rate}"
            )));
        }

        let build_chains = |diffusion: f32, seed: u64| -> Vec<VoiceAllpassChain> {
            (0..n_voices)
                .map(|v| {
                    VoiceAllpassChain::new(
                        config.n_stages,
                        config.max_delay_ms,
                        diffusion,
                        seed,
                        v,
                        sample_rate,
                    )
                })
                .collect()
        };

        // In frequency-dependent mode:
        //   `chains` = low-band chains (reduced diffusion)
        //   `hi_chains` = high-band chains (full diffusion, shifted seed)
        //   `crossovers` = per-voice biquad LP/HP pair at 2 kHz
        //
        // In broadband mode:
        //   `chains` = full-band chains (full diffusion)
        //   `hi_chains` = empty
        //   `crossovers` = None
        let (chains, hi_chains, crossovers) = if config.frequency_dependent {
            let lo_diffusion = config.diffusion * Self::LO_DIFFUSION_SCALE;
            let lo_chains = build_chains(lo_diffusion, config.per_voice_seed);
            let hi_seed = config.per_voice_seed.wrapping_add(0xDEAD_BEEF);
            let hi_chains = build_chains(config.diffusion, hi_seed);

            let xovers: Vec<_> = (0..n_voices)
                .map(|_| {
                    (
                        BiquadFilter::lowpass(Self::CROSSOVER_HZ, sample_rate),
                        BiquadFilter::highpass(Self::CROSSOVER_HZ, sample_rate),
                    )
                })
                .collect();

            (lo_chains, hi_chains, Some(xovers))
        } else {
            let chains = build_chains(config.diffusion, config.per_voice_seed);
            (chains, Vec::new(), None)
        };

        Ok(Self {
            config: config.clone(),
            n_voices,
            sample_rate,
            chains,
            crossovers,
            hi_chains,
        })
    }

    /// Apply per-voice allpass diffusion de-correlation in-place.
    ///
    /// Each voice buffer in `voices` is processed through its unique
    /// allpass chain. The dry/wet mix ratio from the config controls
    /// how much of the decorrelated signal is blended with the original.
    ///
    /// If `voices.len()` does not match `n_voices`, mismatched voices
    /// are left unprocessed (for robustness in streaming scenarios).
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) {
        let mix_wet = self.config.mix;
        let mix_dry = 1.0 - mix_wet;

        // Fast path: no-op if mix is 0 (fully dry).
        if mix_wet < 1e-7 {
            return;
        }

        let n = voices.len().min(self.n_voices);

        if self.crossovers.is_some() {
            // Frequency-dependent mode: split, process bands separately,
            // recombine.
            self.process_frequency_dependent(voices, n, mix_dry, mix_wet);
        } else {
            // Broadband mode: apply allpass chain to full signal.
            self.process_broadband(voices, n, mix_dry, mix_wet);
        }
    }

    /// Broadband processing: apply allpass chain to the full-band signal.
    fn process_broadband(&mut self, voices: &mut [Vec<f32>], n: usize, mix_dry: f32, mix_wet: f32) {
        for voice_idx in 0..n {
            let chain = &mut self.chains[voice_idx];
            for sample in voices[voice_idx].iter_mut() {
                let dry = *sample;
                let wet = chain.process(dry);
                // IEEE 754: 0.0 * NaN = NaN, so guard the mix output.
                let mixed = mix_dry * dry + mix_wet * wet;
                *sample = if mixed.is_finite() { mixed } else { 0.0 };
            }
        }
    }

    /// Frequency-dependent processing: split at 2 kHz, apply different
    /// diffusion amounts to each band, recombine.
    fn process_frequency_dependent(
        &mut self,
        voices: &mut [Vec<f32>],
        n: usize,
        mix_dry: f32,
        mix_wet: f32,
    ) {
        let crossovers = self
            .crossovers
            .as_mut()
            .expect("invariant: crossovers present in frequency_dependent mode");

        for voice_idx in 0..n {
            let (lo_filt, hi_filt) = &mut crossovers[voice_idx];
            let lo_chain = &mut self.chains[voice_idx];
            let hi_chain = &mut self.hi_chains[voice_idx];

            for sample in voices[voice_idx].iter_mut() {
                let dry = *sample;

                // Split into low and high bands.
                let lo = lo_filt.process(dry);
                let hi = hi_filt.process(dry);

                // Apply diffusion to each band with its respective chain.
                let lo_wet = lo_chain.process(lo);
                let hi_wet = hi_chain.process(hi);

                // Recombine bands and apply dry/wet mix.
                let wet = lo_wet + hi_wet;
                // IEEE 754: 0.0 * NaN = NaN, so guard the mix output.
                let mixed = mix_dry * dry + mix_wet * wet;
                *sample = if mixed.is_finite() { mixed } else { 0.0 };
            }
        }
    }

    /// Reset all internal filter state.
    ///
    /// Call this when starting a new utterance or after a discontinuity.
    pub fn reset(&mut self) {
        for chain in &mut self.chains {
            chain.reset();
        }
        for chain in &mut self.hi_chains {
            chain.reset();
        }
        if let Some(ref mut crossovers) = self.crossovers {
            for (lo, hi) in crossovers.iter_mut() {
                lo.reset();
                hi.reset();
            }
        }
    }

    /// Returns the configured number of voices.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.n_voices
    }

    /// Returns the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_decorrelation_tests.rs"]
mod tests;
