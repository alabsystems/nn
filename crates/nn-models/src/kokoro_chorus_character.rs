// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-voice timbral character variation for Kokoro chorus.
//!
//! A real choir sounds rich because each singer has a different body — different
//! vocal tract lengths, different breathiness, different spectral tilt. This
//! module assigns deterministic timbral characteristics to each chorus voice so
//! they sound like distinct people rather than clones of the same singer.
//!
//! # Processing chain per voice
//!
//! ```text
//! Input ──> Vocal tract scaling ──> Breathiness addition ──> Brightness shelf ──> Output
//!           (allpass delay chain)    (filtered noise blend)   (one-pole high-shelf)
//! ```
//!
//! - **Vocal tract scaling** shifts all formants up or down by varying the
//!   effective vocal tract length. A shorter tract (scale < 1.0) shifts formants
//!   up, sounding more childlike; a longer tract (scale > 1.0) shifts formants
//!   down, sounding deeper. Implemented via a cascade of second-order allpass
//!   sections that introduce frequency-dependent group delay, emulating the
//!   acoustic effect of a different-length resonant tube.
//!
//! - **Breathiness** blends band-limited noise (filtered through the voice's
//!   spectral envelope) into voiced regions. More breathiness makes the voice
//!   sound airier and less "perfect."
//!
//! - **Brightness** applies a one-pole high-shelf filter that boosts or cuts
//!   high frequencies. Brighter voices have more presence; darker voices sit
//!   further back in the mix.
//!
//! # Determinism
//!
//! All per-voice parameters are computed deterministically from `seed +
//! voice_index`. Identical seeds produce identical character assignments across
//! runs, which is critical for reproducible audio quality testing and
//! verification.
//!
//! # References
//!
//! - Story, B. H. "Phrase-level speech simulation with an airway modulation
//!   model of speech production." CMBBE: Imaging & Visualization, 2013.
//! - Klatt, D. H. "Software for a cascade/parallel formant synthesizer."
//!   JASA, 67(3), 1980.
//! - Smith, J. O. "Physical Audio Signal Processing."
//!   <https://ccrma.stanford.edu/~jos/pasp/>
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64)
// ---------------------------------------------------------------------------

/// Splitmix64 PRNG — fast, deterministic, excellent avalanche properties.
///
/// Used to derive per-voice characteristics from a seed + voice index.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create seeded from voice index and base seed.
    fn new(seed: u64, voice_index: usize) -> Self {
        // Mix seed with voice index for per-voice diversity.
        let state = seed
            .wrapping_add(voice_index as u64)
            .wrapping_mul(0x9E3779B97F4A7C15);
        Self { state }
    }

    /// Next u64 value.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Random f32 in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Random f32 in [-1.0, 1.0).
    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Character preset
// ---------------------------------------------------------------------------

/// Preset levels for character variation intensity.
///
/// Each preset defines balanced variation amounts that produce musically
/// useful results without requiring manual tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterPreset {
    /// Small variation — sounds like the same singer on different takes.
    /// (tract: 0.05, breathiness: 0.03, brightness: 0.05)
    Subtle,
    /// Medium variation — sounds like similar but distinct voices.
    /// (tract: 0.15, breathiness: 0.08, brightness: 0.15)
    Moderate,
    /// Large variation — sounds like different singers in a choir.
    /// (tract: 0.25, breathiness: 0.15, brightness: 0.25)
    Diverse,
}

impl CharacterPreset {
    /// Convert this preset to a fully configured [`CharacterConfig`].
    #[must_use]
    pub fn to_config(self) -> CharacterConfig {
        match self {
            Self::Subtle => CharacterConfig {
                vocal_tract_variation: 0.05,
                breathiness_variation: 0.03,
                brightness_variation: 0.05,
                seed: 0,
            },
            Self::Moderate => CharacterConfig {
                vocal_tract_variation: 0.15,
                breathiness_variation: 0.08,
                brightness_variation: 0.15,
                seed: 0,
            },
            Self::Diverse => CharacterConfig {
                vocal_tract_variation: 0.25,
                breathiness_variation: 0.15,
                brightness_variation: 0.25,
                seed: 0,
            },
        }
    }

    /// Convert this preset to a [`CharacterConfig`] with a specific seed.
    #[must_use]
    pub fn to_config_with_seed(self, seed: u64) -> CharacterConfig {
        let mut config = self.to_config();
        config.seed = seed;
        config
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for per-voice timbral character variation.
///
/// Each parameter controls how much variation is assigned across chorus
/// voices. The actual per-voice values are derived deterministically from
/// `seed + voice_index`.
///
/// Constructed via [`CharacterConfig::new`] or [`CharacterPreset::to_config`]
/// (required for cross-crate use due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CharacterConfig {
    /// How much to vary vocal tract length across voices.
    ///
    /// Controls the range of formant shifting. 0.0 = no variation (all voices
    /// have identical formants). 0.3 = maximum variation (formants shift
    /// significantly between voices). The actual tract_scale per voice is
    /// computed as `1.0 + variation * random_offset`, bounded to [0.85, 1.15].
    ///
    /// Range: [0.0, 0.3]. Default: 0.15.
    pub vocal_tract_variation: f32,

    /// How much breathiness variation across voices.
    ///
    /// 0.0 = no breathiness added to any voice. 0.2 = some voices get
    /// noticeable breathiness. The actual noise blend amount per voice
    /// is `variation * random_factor`, bounded to [0.0, 0.2].
    ///
    /// Range: [0.0, 0.2]. Default: 0.08.
    pub breathiness_variation: f32,

    /// How much brightness variation across voices.
    ///
    /// Controls the range of high-shelf gain differences. 0.0 = all voices
    /// have the same spectral tilt. 0.3 = voices differ noticeably in
    /// brightness. The actual brightness_db per voice is
    /// `variation * random_offset * 6.0` dB, bounded to [-3.0, 3.0].
    ///
    /// Range: [0.0, 0.3]. Default: 0.15.
    pub brightness_variation: f32,

    /// Deterministic random seed for variation assignment.
    ///
    /// Different seeds produce different voice-to-character mappings while
    /// preserving the overall variation range. Same seed + same voice count
    /// always produces identical assignments.
    pub seed: u64,
}

impl Default for CharacterConfig {
    fn default() -> Self {
        CharacterPreset::Moderate.to_config()
    }
}

impl CharacterConfig {
    /// Create a new character variation configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is non-finite
    /// or outside its valid range.
    pub fn new(
        vocal_tract_variation: f32,
        breathiness_variation: f32,
        brightness_variation: f32,
        seed: u64,
    ) -> Result<Self, KokoroError> {
        let cfg = Self {
            vocal_tract_variation,
            breathiness_variation,
            brightness_variation,
            seed,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Builder: set the vocal tract variation.
    #[must_use]
    pub fn with_vocal_tract(mut self, v: f32) -> Self {
        self.vocal_tract_variation = v;
        self
    }

    /// Builder: set the breathiness variation.
    #[must_use]
    pub fn with_breathiness(mut self, v: f32) -> Self {
        self.breathiness_variation = v;
        self
    }

    /// Builder: set the brightness variation.
    #[must_use]
    pub fn with_brightness(mut self, v: f32) -> Self {
        self.brightness_variation = v;
        self
    }

    /// Builder: set the seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate all fields are within acceptable ranges.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` on out-of-range or non-finite values.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.vocal_tract_variation.is_finite()
            || !(0.0..=0.3).contains(&self.vocal_tract_variation)
        {
            return Err(KokoroError::InvalidConfig {
                field: "vocal_tract_variation",
                reason: format!(
                    "must be finite and in [0.0, 0.3], got {}",
                    self.vocal_tract_variation
                ),
            });
        }
        if !self.breathiness_variation.is_finite()
            || !(0.0..=0.2).contains(&self.breathiness_variation)
        {
            return Err(KokoroError::InvalidConfig {
                field: "breathiness_variation",
                reason: format!(
                    "must be finite and in [0.0, 0.2], got {}",
                    self.breathiness_variation
                ),
            });
        }
        if !self.brightness_variation.is_finite()
            || !(0.0..=0.3).contains(&self.brightness_variation)
        {
            return Err(KokoroError::InvalidConfig {
                field: "brightness_variation",
                reason: format!(
                    "must be finite and in [0.0, 0.3], got {}",
                    self.brightness_variation
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Per-voice character
// ---------------------------------------------------------------------------

/// Timbral characteristics for a single chorus voice.
///
/// Computed deterministically from the config seed and voice index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceCharacter {
    /// Vocal tract length multiplier.
    ///
    /// Values < 1.0 simulate a shorter vocal tract (formants shift up).
    /// Values > 1.0 simulate a longer vocal tract (formants shift down).
    /// Bounded to [0.85, 1.15].
    pub tract_scale: f32,

    /// Noise blend amount for breathiness.
    ///
    /// 0.0 = pure voiced signal. Higher values blend in filtered noise
    /// to simulate breathy phonation. Bounded to [0.0, 0.2].
    pub breathiness: f32,

    /// High-shelf gain adjustment in dB.
    ///
    /// Positive values make the voice brighter (more HF energy).
    /// Negative values make the voice darker. Bounded to [-3.0, 3.0].
    pub brightness_db: f32,
}

impl VoiceCharacter {
    /// Compute the character for a specific voice from the config.
    ///
    /// Voice 0 is always neutral (tract_scale=1.0, breathiness=0.0,
    /// brightness_db=0.0) to serve as the anchor/reference voice.
    #[must_use]
    pub fn from_config(config: &CharacterConfig, voice_index: usize) -> Self {
        if voice_index == 0 {
            return Self {
                tract_scale: 1.0,
                breathiness: 0.0,
                brightness_db: 0.0,
            };
        }

        let mut rng = SplitMix64::new(config.seed, voice_index);

        // Vocal tract: random signed offset scaled by variation amount.
        let tract_offset = rng.next_f32_signed() * config.vocal_tract_variation;
        let tract_scale = (1.0 + tract_offset).clamp(0.85, 1.15);

        // Breathiness: random unsigned amount scaled by variation.
        let breathiness = (rng.next_f32() * config.breathiness_variation).clamp(0.0, 0.2);

        // Brightness: random signed offset in dB.
        let brightness_db =
            (rng.next_f32_signed() * config.brightness_variation * 6.0).clamp(-3.0, 3.0);

        Self {
            tract_scale,
            breathiness,
            brightness_db,
        }
    }

    /// Compute characters for all voices.
    #[must_use]
    pub fn compute_all(config: &CharacterConfig, n_voices: usize) -> Vec<Self> {
        (0..n_voices)
            .map(|i| Self::from_config(config, i))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Allpass vocal tract filter (second-order cascade)
// ---------------------------------------------------------------------------

/// Second-order allpass section for vocal tract length simulation.
///
/// A cascade of these sections introduces frequency-dependent group delay
/// that shifts formant frequencies without changing pitch. The allpass
/// transfer function is:
///
/// ```text
/// H(z) = (a2 + a1*z^-1 + z^-2) / (1 + a1*z^-1 + a2*z^-2)
/// ```
///
/// The filter has unity gain at all frequencies — it only affects phase.
struct AllpassSection {
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl AllpassSection {
    /// Create a new second-order allpass for a given center frequency and
    /// bandwidth, at the specified sample rate.
    fn new(center_hz: f32, bandwidth_hz: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * center_hz / sample_rate;
        let bw = 2.0 * std::f32::consts::PI * bandwidth_hz / sample_rate;
        let cos_w0 = w0.cos();
        // Tangent of half-bandwidth for Q calculation.
        let tan_bw = (bw * 0.5).tan();

        // IEEE 754 guard: if tan_bw is non-finite or near zero, degenerate
        // to pass-through (a1=0, a2=1 gives identity for allpass).
        if !tan_bw.is_finite() || tan_bw.abs() < 1e-12 {
            return Self {
                a1: 0.0,
                a2: 1.0,
                x1: 0.0,
                x2: 0.0,
                y1: 0.0,
                y2: 0.0,
            };
        }

        // Allpass coefficients derived from bandpass prototype.
        let alpha = tan_bw;
        let denom = 1.0 + alpha;
        if !denom.is_finite() || denom.abs() < 1e-12 {
            return Self {
                a1: 0.0,
                a2: 1.0,
                x1: 0.0,
                x2: 0.0,
                y1: 0.0,
                y2: 0.0,
            };
        }
        let a2 = (1.0 - alpha) / denom;
        let a1 = -2.0 * cos_w0 / denom;

        // Final IEEE 754 checks.
        let a1 = if a1.is_finite() { a1 } else { 0.0 };
        let a2 = if a2.is_finite() { a2 } else { 1.0 };

        Self {
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process one sample through this allpass section.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x1 = 0.0;
            self.x2 = 0.0;
            self.y1 = 0.0;
            self.y2 = 0.0;
            return 0.0;
        }

        let y = self.a2 * x + self.a1 * self.x1 + self.x2 - self.a1 * self.y1 - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = if y.is_finite() { y } else { 0.0 };

        self.y1
    }
}

/// Cascade of allpass sections that simulates vocal tract length variation.
///
/// By tuning allpass sections at formant-like center frequencies and varying
/// their bandwidth based on the tract_scale, we create frequency-dependent
/// group delay changes that shift perceived formant positions.
struct VocalTractFilter {
    sections: Vec<AllpassSection>,
}

impl VocalTractFilter {
    /// Formant center frequencies (Hz) for a neutral vocal tract.
    /// These approximate an average adult vocal tract resonance pattern.
    const FORMANT_CENTERS: [f32; 4] = [500.0, 1500.0, 2500.0, 3500.0];

    /// Build a vocal tract filter for the given tract_scale and sample rate.
    ///
    /// tract_scale < 1.0: shorter tract → formants shift up.
    /// tract_scale > 1.0: longer tract → formants shift down.
    fn new(tract_scale: f32, sample_rate: f32) -> Self {
        let nyquist = sample_rate * 0.5;
        let sections = Self::FORMANT_CENTERS
            .iter()
            .filter_map(|&center| {
                // Scale the center frequency inversely to tract length.
                let scaled = center / tract_scale;
                // Skip if above Nyquist.
                if scaled >= nyquist * 0.95 {
                    return None;
                }
                // Bandwidth proportional to center frequency (~10%).
                let bw = scaled * 0.1;
                Some(AllpassSection::new(scaled, bw, sample_rate))
            })
            .collect();
        Self { sections }
    }

    /// Process one sample through the entire cascade.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let mut y = x;
        for section in &mut self.sections {
            y = section.process(y);
        }
        y
    }
}

// ---------------------------------------------------------------------------
// One-pole high-shelf filter for brightness
// ---------------------------------------------------------------------------

/// One-pole high-shelf filter.
///
/// Boosts or cuts frequencies above a crossover frequency.
/// Simple and efficient — one multiply and one add per sample.
///
/// ```text
/// y[n] = (1 - k) * x[n] + k * y[n-1]
/// ```
///
/// where `k` controls the transition and gain_linear adjusts the shelf level.
struct HighShelfFilter {
    /// Shelf gain as a linear multiplier.
    gain: f32,
    /// Filter coefficient controlling crossover frequency.
    coeff: f32,
    /// Previous output (lowpass state).
    y1: f32,
}

impl HighShelfFilter {
    /// Crossover frequency for the high shelf (Hz).
    const CROSSOVER_HZ: f32 = 3000.0;

    /// Create a high-shelf filter for the given brightness_db at sample_rate.
    fn new(brightness_db: f32, sample_rate: f32) -> Self {
        // Convert dB to linear gain.
        let gain = 10.0_f32.powf(brightness_db / 20.0);
        let gain = if gain.is_finite() { gain } else { 1.0 };

        // One-pole lowpass coefficient for crossover frequency.
        let w0 = 2.0 * std::f32::consts::PI * Self::CROSSOVER_HZ / sample_rate;
        let coeff = (-w0).exp();
        let coeff = if coeff.is_finite() { coeff } else { 0.0 };

        Self {
            gain,
            coeff,
            y1: 0.0,
        }
    }

    /// Process one sample.
    ///
    /// Splits into lowpass + highpass, then applies gain to the highpass
    /// component and recombines.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.y1 = 0.0;
            return 0.0;
        }

        // Simple one-pole lowpass.
        let lp = self.coeff * self.y1 + (1.0 - self.coeff) * x;
        self.y1 = if lp.is_finite() { lp } else { 0.0 };

        // Highpass = input - lowpass. Apply gain to highpass, recombine.
        let hp = x - self.y1;
        let out = self.y1 + hp * self.gain;

        if out.is_finite() {
            out
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Main processing function
// ---------------------------------------------------------------------------

/// Apply per-voice character variation to a set of chorus voices.
///
/// Each voice (except voice 0, the anchor) receives a unique timbral
/// character computed deterministically from `config.seed + voice_index`:
///
/// 1. **Vocal tract scaling** — allpass cascade shifts formant frequencies.
/// 2. **Breathiness** — filtered noise blended into the signal.
/// 3. **Brightness** — high-shelf filter adjusts spectral tilt.
///
/// # Arguments
///
/// - `voices`: mutable slice of per-voice audio buffers (mono, f32 samples).
/// - `config`: character variation configuration.
/// - `sample_rate`: audio sample rate in Hz.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if config validation fails, or
/// `KokoroError::InvalidInput` if sample_rate is non-positive or non-finite.
pub fn apply_character_variation(
    voices: &mut [Vec<f32>],
    config: &CharacterConfig,
    sample_rate: f32,
) -> Result<(), KokoroError> {
    config.validate()?;

    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(KokoroError::InvalidInput(format!(
            "sample_rate must be finite and positive, got {sample_rate}"
        )));
    }

    if voices.len() <= 1 {
        return Ok(());
    }

    let characters = VoiceCharacter::compute_all(config, voices.len());

    for (voice_idx, voice) in voices.iter_mut().enumerate() {
        let ch = &characters[voice_idx];

        // Voice 0 is the anchor — no processing.
        if voice_idx == 0 {
            continue;
        }

        // Skip empty voices.
        if voice.is_empty() {
            continue;
        }

        // 1. Vocal tract scaling via allpass cascade.
        if (ch.tract_scale - 1.0).abs() > 1e-6 {
            let mut tract = VocalTractFilter::new(ch.tract_scale, sample_rate);
            for sample in voice.iter_mut() {
                *sample = tract.process(*sample);
            }
        }

        // 2. Breathiness: blend band-limited noise into the signal.
        if ch.breathiness > 1e-6 {
            apply_breathiness(voice, ch.breathiness, config.seed, voice_idx, sample_rate);
        }

        // 3. Brightness: high-shelf filter.
        if ch.brightness_db.abs() > 0.01 {
            let mut shelf = HighShelfFilter::new(ch.brightness_db, sample_rate);
            for sample in voice.iter_mut() {
                *sample = shelf.process(*sample);
            }
        }
    }

    Ok(())
}

/// Blend filtered noise into a voice buffer to simulate breathiness.
///
/// The noise is band-limited to the voice's spectral range using a simple
/// one-pole lowpass at 4kHz (where breath noise concentrates in speech).
/// The blend follows the signal envelope to avoid adding noise to silence.
fn apply_breathiness(
    voice: &mut [f32],
    amount: f32,
    seed: u64,
    voice_index: usize,
    sample_rate: f32,
) {
    let mut rng = SplitMix64::new(seed.wrapping_add(0xB0EA_D000), voice_index);

    // One-pole lowpass coefficient at 4 kHz for band-limiting noise.
    let noise_cutoff = 4000.0_f32.min(sample_rate * 0.45);
    let w0 = 2.0 * std::f32::consts::PI * noise_cutoff / sample_rate;
    let lp_coeff = (-w0).exp();
    let lp_coeff = if lp_coeff.is_finite() { lp_coeff } else { 0.0 };

    let mut noise_state = 0.0_f32;
    // Simple envelope follower for gating noise to voiced regions.
    let mut envelope = 0.0_f32;
    let env_attack = (-2.0 * std::f32::consts::PI * 50.0 / sample_rate).exp();
    let env_release = (-2.0 * std::f32::consts::PI * 10.0 / sample_rate).exp();

    for sample in voice.iter_mut() {
        if !sample.is_finite() {
            continue;
        }

        // Update envelope follower.
        let abs_val = sample.abs();
        if abs_val > envelope {
            envelope = env_attack * envelope + (1.0 - env_attack) * abs_val;
        } else {
            envelope = env_release * envelope + (1.0 - env_release) * abs_val;
        }
        envelope = if envelope.is_finite() { envelope } else { 0.0 };

        // Generate white noise, filter to band-limited.
        let white = rng.next_f32_signed();
        noise_state = lp_coeff * noise_state + (1.0 - lp_coeff) * white;
        noise_state = if noise_state.is_finite() {
            noise_state
        } else {
            0.0
        };

        // Blend noise into signal, gated by envelope.
        let noise_contribution = noise_state * amount * envelope;
        *sample += noise_contribution;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_config_validate_valid() {
        let cfg = CharacterConfig::new(0.1, 0.05, 0.1, 42).unwrap();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_character_config_validate_tract_out_of_range() {
        let result = CharacterConfig::new(0.5, 0.05, 0.1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_character_config_validate_breathiness_out_of_range() {
        let result = CharacterConfig::new(0.1, 0.3, 0.1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_character_config_validate_brightness_out_of_range() {
        let result = CharacterConfig::new(0.1, 0.05, 0.5, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_character_config_validate_nan() {
        let result = CharacterConfig::new(f32::NAN, 0.05, 0.1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_character_config_validate_inf() {
        let result = CharacterConfig::new(0.1, f32::INFINITY, 0.1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_voice_character_anchor_is_neutral() {
        let config = CharacterPreset::Diverse.to_config();
        let ch = VoiceCharacter::from_config(&config, 0);
        assert_eq!(ch.tract_scale, 1.0);
        assert_eq!(ch.breathiness, 0.0);
        assert_eq!(ch.brightness_db, 0.0);
    }

    #[test]
    fn test_voice_character_deterministic() {
        let config = CharacterConfig::new(0.2, 0.1, 0.2, 12345).unwrap();
        let ch1 = VoiceCharacter::from_config(&config, 3);
        let ch2 = VoiceCharacter::from_config(&config, 3);
        assert_eq!(ch1.tract_scale, ch2.tract_scale);
        assert_eq!(ch1.breathiness, ch2.breathiness);
        assert_eq!(ch1.brightness_db, ch2.brightness_db);
    }

    #[test]
    fn test_voice_characters_differ_per_voice() {
        let config = CharacterPreset::Diverse.to_config();
        let chars = VoiceCharacter::compute_all(&config, 5);
        // Voices 1..4 should not all be identical.
        let all_same = chars[1..].windows(2).all(|w| {
            (w[0].tract_scale - w[1].tract_scale).abs() < 1e-9
                && (w[0].breathiness - w[1].breathiness).abs() < 1e-9
                && (w[0].brightness_db - w[1].brightness_db).abs() < 1e-9
        });
        assert!(
            !all_same,
            "non-anchor voices should have distinct characters"
        );
    }

    #[test]
    fn test_voice_character_bounds() {
        let config = CharacterPreset::Diverse.to_config();
        for i in 0..20 {
            let ch = VoiceCharacter::from_config(&config, i);
            assert!(
                (0.85..=1.15).contains(&ch.tract_scale),
                "tract_scale out of bounds: {}",
                ch.tract_scale
            );
            assert!(
                (0.0..=0.2).contains(&ch.breathiness),
                "breathiness out of bounds: {}",
                ch.breathiness
            );
            assert!(
                (-3.0..=3.0).contains(&ch.brightness_db),
                "brightness_db out of bounds: {}",
                ch.brightness_db
            );
        }
    }

    #[test]
    fn test_preset_subtle_less_variation_than_diverse() {
        let subtle = CharacterPreset::Subtle.to_config();
        let diverse = CharacterPreset::Diverse.to_config();
        assert!(subtle.vocal_tract_variation < diverse.vocal_tract_variation);
        assert!(subtle.breathiness_variation < diverse.breathiness_variation);
        assert!(subtle.brightness_variation < diverse.brightness_variation);
    }

    #[test]
    fn test_preset_ordering() {
        let subtle = CharacterPreset::Subtle.to_config();
        let moderate = CharacterPreset::Moderate.to_config();
        let diverse = CharacterPreset::Diverse.to_config();
        assert!(subtle.vocal_tract_variation < moderate.vocal_tract_variation);
        assert!(moderate.vocal_tract_variation < diverse.vocal_tract_variation);
    }

    #[test]
    fn test_apply_character_variation_produces_different_outputs() {
        let sample_rate = 24000.0;
        let n_samples = 4800; // 200ms
        let sine: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
            .collect();

        let mut voices: Vec<Vec<f32>> = (0..4).map(|_| sine.clone()).collect();
        let config = CharacterPreset::Moderate.to_config();
        apply_character_variation(&mut voices, &config, sample_rate).unwrap();

        // Voice 0 should be unchanged (anchor).
        assert_eq!(voices[0], sine, "anchor voice should be unchanged");

        // Voices 1-3 should differ from voice 0 and from each other.
        for i in 1..4 {
            assert_ne!(voices[i], sine, "voice {i} should differ from anchor");
        }
        assert_ne!(voices[1], voices[2], "voice 1 and 2 should differ");
    }

    #[test]
    fn test_apply_character_variation_single_voice() {
        let mut voices = vec![vec![0.5; 100]];
        let config = CharacterPreset::Moderate.to_config();
        let result = apply_character_variation(&mut voices, &config, 24000.0);
        assert!(result.is_ok());
        // Single voice should be unchanged.
        assert_eq!(voices[0], vec![0.5; 100]);
    }

    #[test]
    fn test_apply_character_variation_empty_voices() {
        let mut voices: Vec<Vec<f32>> = vec![vec![], vec![]];
        let config = CharacterPreset::Subtle.to_config();
        let result = apply_character_variation(&mut voices, &config, 24000.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_character_variation_invalid_sample_rate() {
        let mut voices = vec![vec![0.0; 100]; 2];
        let config = CharacterPreset::Subtle.to_config();
        assert!(apply_character_variation(&mut voices, &config, 0.0).is_err());
        assert!(apply_character_variation(&mut voices, &config, -1.0).is_err());
        assert!(apply_character_variation(&mut voices, &config, f32::NAN).is_err());
    }

    #[test]
    fn test_apply_character_variation_no_nan_in_output() {
        let sample_rate = 24000.0;
        let n_samples = 2400;
        let sine: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
            .collect();

        let mut voices: Vec<Vec<f32>> = (0..6).map(|_| sine.clone()).collect();
        let config = CharacterPreset::Diverse.to_config();
        apply_character_variation(&mut voices, &config, sample_rate).unwrap();

        for (vi, voice) in voices.iter().enumerate() {
            for (si, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "NaN/Inf at voice {vi}, sample {si}: {s}");
            }
        }
    }

    #[test]
    fn test_same_seed_produces_identical_results() {
        let sample_rate = 24000.0;
        let sine: Vec<f32> = (0..1200)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
            .collect();

        let config = CharacterConfig::new(0.2, 0.1, 0.2, 42).unwrap();

        let mut voices_a: Vec<Vec<f32>> = (0..3).map(|_| sine.clone()).collect();
        let mut voices_b: Vec<Vec<f32>> = (0..3).map(|_| sine.clone()).collect();

        apply_character_variation(&mut voices_a, &config, sample_rate).unwrap();
        apply_character_variation(&mut voices_b, &config, sample_rate).unwrap();

        for i in 0..3 {
            assert_eq!(
                voices_a[i], voices_b[i],
                "voice {i} should be identical with same seed"
            );
        }
    }

    #[test]
    fn test_different_seeds_produce_different_results() {
        let sample_rate = 24000.0;
        let sine: Vec<f32> = (0..1200)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
            .collect();

        let config_a = CharacterConfig::new(0.2, 0.1, 0.2, 42).unwrap();
        let config_b = CharacterConfig::new(0.2, 0.1, 0.2, 99).unwrap();

        let mut voices_a: Vec<Vec<f32>> = (0..3).map(|_| sine.clone()).collect();
        let mut voices_b: Vec<Vec<f32>> = (0..3).map(|_| sine.clone()).collect();

        apply_character_variation(&mut voices_a, &config_a, sample_rate).unwrap();
        apply_character_variation(&mut voices_b, &config_b, sample_rate).unwrap();

        // Voice 0 (anchor) is the same, but other voices should differ.
        assert_ne!(
            voices_a[1], voices_b[1],
            "different seeds should produce different voice 1"
        );
    }

    #[test]
    fn test_preset_to_config_with_seed() {
        let config = CharacterPreset::Subtle.to_config_with_seed(777);
        assert_eq!(config.seed, 777);
        assert!((config.vocal_tract_variation - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_builder_methods() {
        let config = CharacterConfig::default()
            .with_vocal_tract(0.1)
            .with_breathiness(0.05)
            .with_brightness(0.1)
            .with_seed(42);
        assert!((config.vocal_tract_variation - 0.1).abs() < 1e-6);
        assert!((config.breathiness_variation - 0.05).abs() < 1e-6);
        assert!((config.brightness_variation - 0.1).abs() < 1e-6);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_nan_input_samples_handled() {
        let mut voices = vec![vec![1.0; 100], vec![f32::NAN; 100]];
        let config = CharacterPreset::Moderate.to_config();
        let result = apply_character_variation(&mut voices, &config, 24000.0);
        assert!(result.is_ok());
        // All output samples should be finite (NaN replaced by 0 in filters).
        for &s in &voices[1] {
            assert!(s.is_finite(), "expected finite output, got {s}");
        }
    }
}
