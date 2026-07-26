// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vocal tract resonance modeling for per-voice singer character in chorus.
//!
//! Unlike simple EQ that adjusts broad frequency bands, this module models the
//! resonant cavities of a virtual vocal tract — cascaded formant resonators that
//! shape each voice as if produced by a different singer with a unique body.
//!
//! # Vocal tract model
//!
//! Each voice gets a set of formant resonances (F1-F4) determined by its voice
//! index and the configured variation parameters. The base formant frequencies
//! follow typical adult vocal tract measurements:
//!
//! | Formant | Base Hz | Typical range |
//! |---------|---------|---------------|
//! | F1      | 500     | 270 - 730     |
//! | F2      | 1500    | 840 - 2290    |
//! | F3      | 2500    | 1690 - 3010   |
//! | F4      | 3500    | 3000 - 3700   |
//!
//! Per-voice offsets shift these base frequencies deterministically so that
//! lower voice indices sound deeper (longer tract) and higher indices sound
//! brighter (shorter tract).
//!
//! # Processing chain per voice
//!
//! ```text
//! Input ──> Parallel formant bandpass filters ──> Weighted sum ──>
//!       ──> Nasal anti-resonance (notch @ ~1kHz) ──>
//!       ──> Optional aspiration noise (HP > 3kHz) ──>
//!       ──> Dry/wet mix ──> Output
//! ```
//!
//! # References
//!
//! - Klatt, D. H. "Software for a cascade/parallel formant synthesizer."
//!   JASA, 67(3), 1980.
//! - Fant, G. "Acoustic Theory of Speech Production." Mouton, 1960.
//! - Story, B. H. "Phrase-level speech simulation with an airway modulation
//!   model of speech production." CMBBE: Imaging & Visualization, 2013.
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64)
// ---------------------------------------------------------------------------

/// Splitmix64 PRNG — fast, deterministic, excellent avalanche properties.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64, voice_index: usize) -> Self {
        let state = seed
            .wrapping_add(voice_index as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Biquad filter (direct-form II transposed)
// ---------------------------------------------------------------------------

/// Second-order biquad filter for formant resonance and anti-resonance.
#[derive(Debug, Clone)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Bandpass filter centered at `freq_hz` with quality factor `q`.
    fn bandpass(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let omega = std::f32::consts::TAU * (freq_hz / sample_rate);
        let (sin_w, cos_w) = (omega.sin(), omega.cos());
        let alpha = sin_w / (2.0 * q);
        let a0_inv = 1.0 / (1.0 + alpha);
        Self {
            b0: alpha * a0_inv,
            b1: 0.0,
            b2: -alpha * a0_inv,
            a1: (-2.0 * cos_w) * a0_inv,
            a2: (1.0 - alpha) * a0_inv,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Notch (band-reject) filter centered at `freq_hz` with quality factor `q`.
    fn notch(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let omega = std::f32::consts::TAU * (freq_hz / sample_rate);
        let (sin_w, cos_w) = (omega.sin(), omega.cos());
        let alpha = sin_w / (2.0 * q);
        let a0_inv = 1.0 / (1.0 + alpha);
        Self {
            b0: a0_inv,
            b1: (-2.0 * cos_w) * a0_inv,
            b2: a0_inv,
            a1: (-2.0 * cos_w) * a0_inv,
            a2: (1.0 - alpha) * a0_inv,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// High-pass filter at `freq_hz` (Butterworth Q = 1/sqrt(2)).
    fn highpass(freq_hz: f32, sample_rate: f32) -> Self {
        let omega = std::f32::consts::TAU * (freq_hz / sample_rate);
        let (sin_w, cos_w) = (omega.sin(), omega.cos());
        let alpha = sin_w / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0_inv = 1.0 / (1.0 + alpha);
        Self {
            b0: f32::midpoint(1.0, cos_w) * a0_inv,
            b1: -(1.0 + cos_w) * a0_inv,
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
// Constants
// ---------------------------------------------------------------------------

/// Base formant frequencies (Hz) for average adult vocal tract.
const BASE_FORMANTS: [f32; 4] = [500.0, 1500.0, 2500.0, 3500.0];

/// Typical formant frequency ranges: (min, max) in Hz.
const FORMANT_RANGES: [(f32, f32); 4] = [
    (270.0, 730.0),   // F1
    (840.0, 2290.0),  // F2
    (1690.0, 3010.0), // F3
    (3000.0, 3700.0), // F4
];

/// Base Q (quality factor) for each formant resonator.
const BASE_Q: [f32; 4] = [5.0, 8.0, 10.0, 12.0];

/// Relative amplitude weighting: F1 strongest, F4 weakest.
const FORMANT_WEIGHTS: [f32; 4] = [1.0, 0.7, 0.4, 0.2];

/// Nasal anti-resonance center frequency (Hz).
const NASAL_NOTCH_HZ: f32 = 1000.0;

/// Aspiration noise high-pass cutoff (Hz).
const ASPIRATION_HP_HZ: f32 = 3000.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for vocal tract resonance modeling.
///
/// Controls how much each voice's formant structure varies, the number of
/// resonances modeled, and additional spectral shaping parameters.
///
/// Constructed via [`VocalTractConfig::new`] or preset methods
/// ([`natural`](VocalTractConfig::natural),
///  [`choir`](VocalTractConfig::choir), etc.).
/// Required for cross-crate use due to `#[non_exhaustive]`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VocalTractConfig {
    /// Number of chorus voices.
    pub n_voices: usize,

    /// How different each voice's vocal tract is (0.0-1.0).
    ///
    /// At 0.0 all voices share identical formants; at 1.0 formants span
    /// the full physiological range. Default: 0.3.
    pub tract_variation: f32,

    /// Number of formant resonances to model (1-4, maps to F1-F4).
    /// Default: 4.
    pub formant_count: usize,

    /// Nasal resonance contribution (0.0-1.0). Controls the depth of a
    /// notch anti-resonance around 1000 Hz. Default: 0.1.
    pub nasal_amount: f32,

    /// Lower formant emphasis (0.0-1.0). Higher values increase the
    /// relative weight of F1/F2. Default: 0.5.
    pub throat_depth: f32,

    /// Higher formant emphasis (0.0-1.0). Higher values increase the
    /// relative weight of F3/F4. Default: 0.5.
    pub brightness: f32,

    /// Aspiration noise mix (0.0-1.0). Blends filtered white noise
    /// above 3 kHz to emulate breathy voice quality. Default: 0.05.
    pub breathiness: f32,

    /// Formant frequency spread between voices (0.0-1.0). At 0.0 all
    /// voices share the same formant offset; at 1.0 the range spans
    /// from bass to soprano. Default: 0.2.
    pub gender_spread: f32,

    /// Dry/wet mix (0.0 = fully dry, 1.0 = fully wet). Default: 0.4.
    pub mix: f32,
}

impl Default for VocalTractConfig {
    fn default() -> Self {
        Self {
            n_voices: 4,
            tract_variation: 0.3,
            formant_count: 4,
            nasal_amount: 0.1,
            throat_depth: 0.5,
            brightness: 0.5,
            breathiness: 0.05,
            gender_spread: 0.2,
            mix: 0.4,
        }
    }
}

impl VocalTractConfig {
    /// Create a new vocal tract configuration with explicit parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is non-finite
    /// or outside its valid range.
    pub fn new(n_voices: usize) -> Result<Self, KokoroError> {
        let cfg = Self {
            n_voices,
            ..Default::default()
        };
        cfg.validate()?;
        Ok(cfg)
    }

    // -- Builder methods -------------------------------------------------------

    /// Builder: set tract variation amount.
    #[must_use]
    pub fn with_tract_variation(mut self, v: f32) -> Self {
        self.tract_variation = v;
        self
    }

    /// Builder: set formant count.
    #[must_use]
    pub fn with_formant_count(mut self, n: usize) -> Self {
        self.formant_count = n;
        self
    }

    /// Builder: set nasal amount.
    #[must_use]
    pub fn with_nasal_amount(mut self, v: f32) -> Self {
        self.nasal_amount = v;
        self
    }

    /// Builder: set throat depth.
    #[must_use]
    pub fn with_throat_depth(mut self, v: f32) -> Self {
        self.throat_depth = v;
        self
    }

    /// Builder: set brightness.
    #[must_use]
    pub fn with_brightness(mut self, v: f32) -> Self {
        self.brightness = v;
        self
    }

    /// Builder: set breathiness.
    #[must_use]
    pub fn with_breathiness(mut self, v: f32) -> Self {
        self.breathiness = v;
        self
    }

    /// Builder: set gender spread.
    #[must_use]
    pub fn with_gender_spread(mut self, v: f32) -> Self {
        self.gender_spread = v;
        self
    }

    /// Builder: set dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, v: f32) -> Self {
        self.mix = v;
        self
    }

    // -- Presets ---------------------------------------------------------------

    /// Subtle variation, minimal breathiness — sounds like the same choir
    /// section with natural individuality.
    #[must_use]
    pub fn natural(n_voices: usize) -> Self {
        Self {
            n_voices,
            tract_variation: 0.15,
            formant_count: 4,
            nasal_amount: 0.05,
            throat_depth: 0.5,
            brightness: 0.5,
            breathiness: 0.02,
            gender_spread: 0.1,
            mix: 0.35,
        }
    }

    /// Moderate variation, slight breathiness, wider gender spread —
    /// sounds like a real choir with distinct singers.
    #[must_use]
    pub fn choir(n_voices: usize) -> Self {
        Self {
            n_voices,
            tract_variation: 0.35,
            formant_count: 4,
            nasal_amount: 0.12,
            throat_depth: 0.5,
            brightness: 0.55,
            breathiness: 0.08,
            gender_spread: 0.35,
            mix: 0.5,
        }
    }

    /// Large variation, each voice very distinct — dramatic choir with
    /// audibly different vocal characters.
    #[must_use]
    pub fn extreme(n_voices: usize) -> Self {
        Self {
            n_voices,
            tract_variation: 0.7,
            formant_count: 4,
            nasal_amount: 0.2,
            throat_depth: 0.4,
            brightness: 0.6,
            breathiness: 0.15,
            gender_spread: 0.6,
            mix: 0.6,
        }
    }

    /// Minimal variation, voices sound nearly identical — tight unison
    /// with just enough resonance variation to avoid comb filtering.
    #[must_use]
    pub fn unison(n_voices: usize) -> Self {
        Self {
            n_voices,
            tract_variation: 0.05,
            formant_count: 3,
            nasal_amount: 0.02,
            throat_depth: 0.5,
            brightness: 0.5,
            breathiness: 0.01,
            gender_spread: 0.03,
            mix: 0.2,
        }
    }

    // -- Validation ------------------------------------------------------------

    /// Validate all fields are within acceptable ranges.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` on out-of-range or non-finite values.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.n_voices == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: "must be >= 1".to_string(),
            });
        }
        check_unit("tract_variation", self.tract_variation)?;
        if !(1..=4).contains(&self.formant_count) {
            return Err(KokoroError::InvalidConfig {
                field: "formant_count",
                reason: format!("must be in [1, 4], got {}", self.formant_count),
            });
        }
        check_unit("nasal_amount", self.nasal_amount)?;
        check_unit("throat_depth", self.throat_depth)?;
        check_unit("brightness", self.brightness)?;
        check_unit("breathiness", self.breathiness)?;
        check_unit("gender_spread", self.gender_spread)?;
        check_unit("mix", self.mix)?;
        Ok(())
    }
}

/// Validate a float parameter is finite and in [0.0, 1.0].
fn check_unit(name: &'static str, val: f32) -> Result<(), KokoroError> {
    if !val.is_finite() || !(0.0..=1.0).contains(&val) {
        return Err(KokoroError::InvalidConfig {
            field: name,
            reason: format!("must be finite and in [0.0, 1.0], got {val}"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-voice vocal tract model
// ---------------------------------------------------------------------------

/// Formant parameters for a single resonance.
#[derive(Debug, Clone)]
pub struct FormantParams {
    /// Center frequency in Hz.
    pub freq_hz: f32,
    /// Quality factor (bandwidth control).
    pub q: f32,
    /// Relative amplitude weight.
    pub weight: f32,
}

/// Per-voice resonance model: cascaded formant bandpass filters + nasal notch.
#[derive(Debug, Clone)]
pub struct VocalTractModel {
    /// Formant bandpass filters (up to 4).
    formants: Vec<Biquad>,
    /// Per-formant amplitude weights (adjusted for throat_depth / brightness).
    weights: Vec<f32>,
    /// Nasal anti-resonance notch filter.
    nasal_notch: Biquad,
    /// Nasal notch depth (0.0 = no notch, 1.0 = full notch).
    nasal_depth: f32,
    /// Aspiration noise high-pass filter.
    aspiration_hp: Biquad,
    /// Aspiration noise amount (0.0-1.0).
    aspiration_amount: f32,
    /// Simple noise generator state (LCG).
    noise_state: u32,
    /// The formant parameters used to build this model.
    params: Vec<FormantParams>,
}

impl VocalTractModel {
    /// Query the formant parameters for this voice.
    pub fn formant_params(&self) -> &[FormantParams] {
        &self.params
    }

    /// Generate a pseudo-random noise sample in [-1.0, 1.0].
    #[inline]
    fn noise(&mut self) -> f32 {
        // Linear congruential generator — fast, adequate for aspiration noise.
        self.noise_state = self
            .noise_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.noise_state as i32) as f32 / i32::MAX as f32
    }

    fn reset(&mut self) {
        for f in &mut self.formants {
            f.reset();
        }
        self.nasal_notch.reset();
        self.aspiration_hp.reset();
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Vocal tract resonance processor that applies per-voice formant shaping.
///
/// Created from a [`VocalTractConfig`] and a sample rate. Each call to
/// [`process_voices`](Self::process_voices) applies the resonance model
/// in-place to the provided audio buffers.
#[derive(Debug, Clone)]
pub struct VocalTractProcessor {
    config: VocalTractConfig,
    models: Vec<VocalTractModel>,
    sample_rate: f32,
}

impl VocalTractProcessor {
    /// Create a new processor with the given configuration and sample rate.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the configuration is invalid
    /// or the sample rate is non-positive/non-finite.
    pub fn new(config: &VocalTractConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be finite and positive, got {sample_rate}"),
            });
        }

        let models = (0..config.n_voices)
            .map(|vi| build_voice_model(config, vi, sample_rate))
            .collect();

        Ok(Self {
            config: config.clone(),
            models,
            sample_rate,
        })
    }

    /// Process all voices in-place. The slice length must equal `n_voices`.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the number of voice buffers
    /// does not match `n_voices`.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) -> Result<(), KokoroError> {
        if voices.len() != self.config.n_voices {
            return Err(KokoroError::InvalidConfig {
                field: "voices",
                reason: format!(
                    "expected {} voice buffers, got {}",
                    self.config.n_voices,
                    voices.len()
                ),
            });
        }
        for (i, buf) in voices.iter_mut().enumerate() {
            self.process_voice(buf, i);
        }
        Ok(())
    }

    /// Process a single voice buffer in-place.
    ///
    /// `voice_index` must be in `0..n_voices`. Out-of-range indices are
    /// silently ignored (the buffer is returned unmodified).
    pub fn process_voice(&mut self, audio: &mut [f32], voice_index: usize) {
        let model = match self.models.get_mut(voice_index) {
            Some(m) => m,
            None => return,
        };
        let mix = self.config.mix;
        if mix <= 0.0 || audio.is_empty() {
            return;
        }

        for sample in audio.iter_mut() {
            let dry = *sample;

            // -- Parallel formant resonators ----------------------------------
            let mut wet = 0.0f32;
            for (filt, &w) in model.formants.iter_mut().zip(model.weights.iter()) {
                wet += filt.process(dry) * w;
            }

            // Normalize by total weight to prevent amplitude inflation.
            let total_w: f32 = model.weights.iter().sum();
            if total_w > 0.0 {
                wet /= total_w;
            }

            // -- Nasal anti-resonance -----------------------------------------
            if model.nasal_depth > 0.0 {
                let notched = model.nasal_notch.process(wet);
                wet = wet * (1.0 - model.nasal_depth) + notched * model.nasal_depth;
            }

            // -- Aspiration noise ---------------------------------------------
            if model.aspiration_amount > 0.0 {
                let noise_raw = model.noise();
                let noise_hp = model.aspiration_hp.process(noise_raw);
                // Scale noise relative to signal amplitude for naturalness.
                let env = dry.abs().min(1.0);
                wet += noise_hp * model.aspiration_amount * env;
            }

            // -- Dry/wet blend ------------------------------------------------
            *sample = dry * (1.0 - mix) + wet * mix;

            // Defense-in-depth: clamp non-finite results.
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }

    /// Reset all filter states (e.g. between non-contiguous audio chunks).
    pub fn reset(&mut self) {
        for model in &mut self.models {
            model.reset();
        }
    }

    /// Access the underlying vocal tract models (for inspection/testing).
    pub fn models(&self) -> &[VocalTractModel] {
        &self.models
    }

    /// The sample rate this processor was configured for.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Voice model construction
// ---------------------------------------------------------------------------

/// Compute per-voice formant offset based on voice index.
///
/// Lower voice indices get negative offsets (deeper), higher indices get
/// positive offsets (brighter). The magnitude is controlled by
/// `tract_variation * gender_spread`.
fn voice_seed_offset(
    voice_index: usize,
    n_voices: usize,
    tract_variation: f32,
    gender_spread: f32,
) -> f32 {
    if n_voices <= 1 {
        return 0.0;
    }
    // Linear spread: voice 0 gets most negative, voice (n-1) gets most positive.
    let t = voice_index as f32 / (n_voices - 1) as f32; // 0.0 .. 1.0
    let centered = t * 2.0 - 1.0; // -1.0 .. 1.0
    centered * tract_variation * gender_spread
}

/// Build the vocal tract model for a single voice.
fn build_voice_model(
    config: &VocalTractConfig,
    voice_index: usize,
    sample_rate: f32,
) -> VocalTractModel {
    let n_formants = config.formant_count.min(4);
    let base_offset = voice_seed_offset(
        voice_index,
        config.n_voices,
        config.tract_variation,
        config.gender_spread,
    );

    // Per-voice jitter via PRNG for formant frequency and Q randomization.
    let mut rng = SplitMix64::new(0xCAFE_BABE_DEAD_BEEF, voice_index);

    let mut formants = Vec::with_capacity(n_formants);
    let mut weights = Vec::with_capacity(n_formants);
    let mut params = Vec::with_capacity(n_formants);

    for i in 0..n_formants {
        let base_freq = BASE_FORMANTS[i];
        let (range_lo, range_hi) = FORMANT_RANGES[i];

        // Per-formant jitter: small random perturbation so voices aren't
        // perfectly linearly spaced.
        let jitter = rng.next_f32_signed() * config.tract_variation * 0.1;

        let freq = base_freq * (1.0 + base_offset + jitter);
        let freq = freq.clamp(range_lo, range_hi);

        // Q varies slightly per voice for natural feel.
        let q_jitter = 1.0 + rng.next_f32_signed() * config.tract_variation * 0.2;
        let q = (BASE_Q[i] * q_jitter).clamp(2.0, 20.0);

        // Weight: adjust by throat_depth (boosts F1/F2) and brightness (boosts F3/F4).
        let mut w = FORMANT_WEIGHTS[i];
        if i < 2 {
            w *= 0.7 + config.throat_depth * 0.6; // [0.7, 1.3]
        } else {
            w *= 0.7 + config.brightness * 0.6;
        }

        formants.push(Biquad::bandpass(freq, q, sample_rate));
        weights.push(w);
        params.push(FormantParams {
            freq_hz: freq,
            q,
            weight: w,
        });
    }

    // Nasal anti-resonance: notch around 1000 Hz with per-voice variation.
    let nasal_freq_jitter = 1.0 + rng.next_f32_signed() * 0.05;
    let nasal_freq = NASAL_NOTCH_HZ * nasal_freq_jitter;
    let nasal_q = 3.0 + rng.next_f32() * 2.0; // Q in [3, 5]
    let nasal_notch = Biquad::notch(nasal_freq, nasal_q, sample_rate);

    let aspiration_hp = Biquad::highpass(ASPIRATION_HP_HZ, sample_rate);

    VocalTractModel {
        formants,
        weights,
        nasal_notch,
        nasal_depth: config.nasal_amount,
        aspiration_hp,
        aspiration_amount: config.breathiness,
        noise_state: 0x1234_5678_u32.wrapping_add(voice_index as u32 * 0x9E37),
        params,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let cfg = VocalTractConfig::default();
        cfg.validate().expect("default config should be valid");
    }

    #[test]
    fn test_preset_natural_validates() {
        let cfg = VocalTractConfig::natural(4);
        cfg.validate().expect("natural preset should be valid");
    }

    #[test]
    fn test_preset_choir_validates() {
        let cfg = VocalTractConfig::choir(6);
        cfg.validate().expect("choir preset should be valid");
    }

    #[test]
    fn test_preset_extreme_validates() {
        let cfg = VocalTractConfig::extreme(8);
        cfg.validate().expect("extreme preset should be valid");
    }

    #[test]
    fn test_preset_unison_validates() {
        let cfg = VocalTractConfig::unison(3);
        cfg.validate().expect("unison preset should be valid");
    }

    #[test]
    fn test_builder_chain() {
        let cfg = VocalTractConfig::new(4)
            .expect("base config should be valid")
            .with_tract_variation(0.5)
            .with_formant_count(3)
            .with_nasal_amount(0.2)
            .with_breathiness(0.1)
            .with_mix(0.6);
        cfg.validate()
            .expect("builder chain should produce valid config");
        assert_eq!(cfg.formant_count, 3);
        assert!((cfg.mix - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_config_rejected() {
        let cfg = VocalTractConfig::default().with_mix(1.5);
        assert!(cfg.validate().is_err());

        let cfg = VocalTractConfig::default().with_tract_variation(-0.1);
        assert!(cfg.validate().is_err());

        let cfg = VocalTractConfig::default().with_formant_count(0);
        assert!(cfg.validate().is_err());

        let cfg = VocalTractConfig::default().with_formant_count(5);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nan_config_rejected() {
        let cfg = VocalTractConfig::default().with_mix(f32::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_zero_voices_rejected() {
        assert!(VocalTractConfig::new(0).is_err());
    }

    #[test]
    fn test_processor_creation() {
        let cfg = VocalTractConfig::natural(4);
        let proc = VocalTractProcessor::new(&cfg, 24000.0);
        assert!(proc.is_ok());
        let proc = proc.unwrap();
        assert_eq!(proc.models().len(), 4);
    }

    #[test]
    fn test_processor_invalid_sample_rate() {
        let cfg = VocalTractConfig::natural(2);
        assert!(VocalTractProcessor::new(&cfg, 0.0).is_err());
        assert!(VocalTractProcessor::new(&cfg, -44100.0).is_err());
        assert!(VocalTractProcessor::new(&cfg, f32::NAN).is_err());
    }

    #[test]
    fn test_process_voices_length_mismatch() {
        let cfg = VocalTractConfig::natural(3);
        let mut proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();
        let mut voices = vec![vec![0.0f32; 100], vec![0.0; 100]]; // only 2, not 3
        assert!(proc.process_voices(&mut voices).is_err());
    }

    #[test]
    fn test_process_voice_modifies_audio() {
        let cfg = VocalTractConfig::choir(2);
        let mut proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();

        // Generate a 500 Hz sine wave (near F1) — formant filter should pass it.
        let sr = 24000.0;
        let n = 2400; // 100ms
        let mut audio: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * 500.0 * i as f32 / sr).sin() * 0.5)
            .collect();
        let original = audio.clone();

        proc.process_voice(&mut audio, 0);

        // Audio should be modified (not identical to dry signal).
        let changed = audio
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "vocal tract processing should modify the audio");
    }

    #[test]
    fn test_process_preserves_silence() {
        let cfg = VocalTractConfig::natural(2);
        let mut proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();

        let mut silence = vec![0.0f32; 1000];
        proc.process_voice(&mut silence, 0);

        // With zero input, formant filters produce zero (after initial transient).
        // Aspiration noise is scaled by envelope (abs of dry signal), so it's 0 too.
        let max_val = silence.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 1e-4,
            "silence in should produce near-silence out, got max {max_val}"
        );
    }

    #[test]
    fn test_formants_differ_between_voices() {
        let cfg = VocalTractConfig::choir(4);
        let proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();

        // All voices should have different F1 frequencies.
        let f1_freqs: Vec<f32> = proc
            .models()
            .iter()
            .map(|m| m.formant_params()[0].freq_hz)
            .collect();

        for i in 0..f1_freqs.len() {
            for j in (i + 1)..f1_freqs.len() {
                assert!(
                    (f1_freqs[i] - f1_freqs[j]).abs() > 0.1,
                    "voices {i} and {j} should have different F1: {} vs {}",
                    f1_freqs[i],
                    f1_freqs[j],
                );
            }
        }
    }

    #[test]
    fn test_lower_voice_has_lower_formants() {
        let cfg = VocalTractConfig::choir(4);
        let proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();

        let f1_first = proc.models()[0].formant_params()[0].freq_hz;
        let f1_last = proc.models()[3].formant_params()[0].freq_hz;

        // Voice 0 should have lower formants than voice 3.
        assert!(
            f1_first < f1_last,
            "voice 0 F1 ({f1_first}) should be lower than voice 3 F1 ({f1_last})"
        );
    }

    #[test]
    fn test_formants_within_range() {
        let cfg = VocalTractConfig::extreme(8);
        let proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();

        for (vi, model) in proc.models().iter().enumerate() {
            for (fi, p) in model.formant_params().iter().enumerate() {
                let (lo, hi) = FORMANT_RANGES[fi];
                assert!(
                    p.freq_hz >= lo && p.freq_hz <= hi,
                    "voice {vi} formant F{} = {} Hz outside range [{lo}, {hi}]",
                    fi + 1,
                    p.freq_hz,
                );
            }
        }
    }

    #[test]
    fn test_no_nan_in_output() {
        let cfg = VocalTractConfig::extreme(4);
        let mut proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();

        // Feed a loud signal with some edge values.
        let mut audio: Vec<f32> = (0..4800)
            .map(|i| {
                let t = i as f32 / 24000.0;
                (std::f32::consts::TAU * 440.0 * t).sin() * 0.9
            })
            .collect();

        proc.process_voice(&mut audio, 0);

        assert!(
            audio.iter().all(|x| x.is_finite()),
            "output must contain no NaN/Inf values"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = VocalTractConfig::natural(2);
        let mut proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();

        // Process some audio to fill filter state.
        let mut audio: Vec<f32> = (0..1200)
            .map(|i| (std::f32::consts::TAU * 300.0 * i as f32 / 24000.0).sin())
            .collect();
        proc.process_voice(&mut audio, 0);

        // Reset and process silence — should converge to zero quickly.
        proc.reset();
        let mut silence = vec![0.0f32; 1000];
        proc.process_voice(&mut silence, 0);

        let max_val = silence.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 1e-3,
            "after reset, silence should produce near-silence, got max {max_val}"
        );
    }

    #[test]
    fn test_mix_zero_passes_through() {
        let cfg = VocalTractConfig::natural(2).with_mix(0.0);
        let mut proc = VocalTractProcessor::new(&cfg, 24000.0).unwrap();

        let original: Vec<f32> = (0..1000)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let mut audio = original.clone();
        proc.process_voice(&mut audio, 0);

        assert_eq!(audio, original, "mix=0.0 should pass through unchanged");
    }
}
