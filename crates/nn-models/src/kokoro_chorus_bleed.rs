// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Voice bleed (crosstalk) and proximity effect for natural chorus microphone
//! simulation.
//!
//! In a real choir recording, every microphone picks up some signal from
//! neighboring singers. This "bleed" or "crosstalk" is a defining
//! characteristic of ensemble recordings — it glues voices together by
//! creating subtle shared ambience. Without it, multi-voice synthesis
//! sounds unnaturally isolated, like each singer is in a separate room.
//!
//! # Bleed model
//!
//! For each voice pair (i, j), the bleed amount is:
//!
//! ```text
//! bleed(i, j) = bleed_amount * (1.0 / (1.0 + proximity_rolloff * distance(i, j)))
//! ```
//!
//! where `distance(i, j) = |i - j| / n_voices` normalizes voice separation
//! to [0, 1]. The bleed signal is lowpass-filtered before addition, simulating
//! the fact that microphone crosstalk is dominated by low and mid frequencies
//! (high frequencies are more directional and attenuate faster with distance).
//!
//! # Proximity effect
//!
//! Close-miked sources exhibit bass boost from the pressure gradient
//! response of cardioid microphones. Voice[0] (the "closest" voice) gets
//! the most bass boost, rolling off linearly across voice indices so
//! voice[n-1] gets none. This is implemented as a simple one-pole shelving
//! filter applied per voice.
//!
//! # Placement in the chorus pipeline
//!
//! Bleed is applied **after** per-voice processing (detuning, vibrato,
//! humanization, EQ) and **before** stereo panning and reverb:
//!
//! ```text
//! Per-voice processing -> bleed/crosstalk -> stereo mix -> reverb -> master
//! ```
//!
//! This ordering is deliberate: each voice should already have its unique
//! character before bleed mixes small amounts between neighbors.
//!
//! # References
//!
//! - Eargle, J. "Handbook of Recording Engineering." 4th ed., Springer, 2005.
//!   Chapter 5: Microphone Placement and Crosstalk.
//! - Streicher, R. & Everest, F. A. "The New Stereo Soundbook." 3rd ed.,
//!   Audio Engineering Associates, 2006.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Deterministic PRNG (LCG)
// ---------------------------------------------------------------------------

/// Simple linear congruential generator for deterministic per-voice phase
/// variation. Same design as other chorus modules (humanize, breath).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(voice_index: usize, salt: u64) -> Self {
        let seed = (voice_index as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(salt)
            .wrapping_add(1);
        Self { state: seed }
    }

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
}

// ---------------------------------------------------------------------------
// BleedConfig
// ---------------------------------------------------------------------------

/// Configuration for voice bleed (microphone crosstalk) simulation.
///
/// Constructed via [`BleedConfig::new`] (required for cross-crate use due
/// to `#[non_exhaustive]`). Builder methods allow chaining adjustments.
///
/// # Defaults
///
/// | Parameter | Default | Range |
/// |-----------|---------|-------|
/// | `bleed_amount` | 0.05 | [0.0, 0.15] |
/// | `proximity_rolloff` | 2.0 | [0.5, 4.0] |
/// | `lowpass_freq` | 2000.0 | [800.0, 4000.0] |
/// | `seed` | 0 | any u64 |
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BleedConfig {
    /// Fraction of each voice that bleeds into adjacent voices.
    ///
    /// Higher values create more cohesive (but less separated) voices.
    /// Must be in [0.0, 0.15] and finite. Typical values: 0.03-0.08.
    pub bleed_amount: f32,

    /// How quickly bleed attenuates with voice distance.
    ///
    /// Higher values mean bleed drops off faster with distance, so only
    /// immediate neighbors contribute significantly. Must be in [0.5, 4.0]
    /// and finite.
    pub proximity_rolloff: f32,

    /// Cutoff frequency (Hz) for the one-pole lowpass applied to bleed signal.
    ///
    /// Simulates the fact that microphone crosstalk is dominated by low/mid
    /// frequencies. Must be in [800.0, 4000.0] and finite.
    pub lowpass_freq: f32,

    /// Deterministic seed for per-voice phase variation.
    ///
    /// Different seeds produce slightly different per-voice lowpass initial
    /// conditions, adding subtle variety to the bleed character.
    pub seed: u64,
}

impl Default for BleedConfig {
    fn default() -> Self {
        Self {
            bleed_amount: 0.05,
            proximity_rolloff: 2.0,
            lowpass_freq: 2000.0,
            seed: 0,
        }
    }
}

impl BleedConfig {
    /// Create a new bleed configuration with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the bleed amount (fraction of signal that crosses into neighbors).
    #[must_use]
    pub fn with_bleed_amount(mut self, amount: f32) -> Self {
        self.bleed_amount = amount;
        self
    }

    /// Set the proximity rolloff (higher = faster attenuation with distance).
    #[must_use]
    pub fn with_proximity_rolloff(mut self, rolloff: f32) -> Self {
        self.proximity_rolloff = rolloff;
        self
    }

    /// Set the lowpass cutoff frequency for bleed filtering.
    #[must_use]
    pub fn with_lowpass_freq(mut self, freq: f32) -> Self {
        self.lowpass_freq = freq;
        self
    }

    /// Set the deterministic seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate that all parameters are within valid ranges.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range
    /// or non-finite.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.bleed_amount.is_finite() || !(0.0..=0.15).contains(&self.bleed_amount) {
            return Err(KokoroError::InvalidConfig {
                field: "bleed_amount",
                reason: format!(
                    "must be finite and in [0.0, 0.15], got {}",
                    self.bleed_amount,
                ),
            });
        }
        if !self.proximity_rolloff.is_finite() || !(0.5..=4.0).contains(&self.proximity_rolloff) {
            return Err(KokoroError::InvalidConfig {
                field: "proximity_rolloff",
                reason: format!(
                    "must be finite and in [0.5, 4.0], got {}",
                    self.proximity_rolloff,
                ),
            });
        }
        if !self.lowpass_freq.is_finite() || !(800.0..=4000.0).contains(&self.lowpass_freq) {
            return Err(KokoroError::InvalidConfig {
                field: "lowpass_freq",
                reason: format!(
                    "must be finite and in [800.0, 4000.0], got {}",
                    self.lowpass_freq,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ProximityEffect
// ---------------------------------------------------------------------------

/// Configuration for proximity-effect bass boost across voices.
///
/// Close-miked cardioid microphones exhibit a bass boost (the proximity
/// effect). Voice[0] is treated as the closest voice and gets the most
/// boost; voice[n-1] gets none. The boost is applied as a one-pole
/// shelving filter per voice.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ProximityEffect {
    /// Bass boost in dB for the closest voice (voice[0]).
    ///
    /// Must be in [0.0, 6.0] and finite. Default: 2.0 dB.
    pub proximity_boost_db: f32,

    /// Shelving filter cutoff frequency in Hz.
    ///
    /// Frequencies below this get boosted; frequencies above are unaffected.
    /// Must be in [80.0, 500.0] and finite. Default: 250.0 Hz.
    pub shelf_freq: f32,
}

impl Default for ProximityEffect {
    fn default() -> Self {
        Self {
            proximity_boost_db: 2.0,
            shelf_freq: 250.0,
        }
    }
}

impl ProximityEffect {
    /// Create a new proximity effect configuration with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the bass boost amount in dB.
    #[must_use]
    pub fn with_boost_db(mut self, db: f32) -> Self {
        self.proximity_boost_db = db;
        self
    }

    /// Set the shelving filter cutoff frequency.
    #[must_use]
    pub fn with_shelf_freq(mut self, freq: f32) -> Self {
        self.shelf_freq = freq;
        self
    }

    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.proximity_boost_db.is_finite() || !(0.0..=6.0).contains(&self.proximity_boost_db) {
            return Err(KokoroError::InvalidConfig {
                field: "proximity_boost_db",
                reason: format!(
                    "must be finite and in [0.0, 6.0], got {}",
                    self.proximity_boost_db,
                ),
            });
        }
        if !self.shelf_freq.is_finite() || !(80.0..=500.0).contains(&self.shelf_freq) {
            return Err(KokoroError::InvalidConfig {
                field: "shelf_freq",
                reason: format!(
                    "must be finite and in [80.0, 500.0], got {}",
                    self.shelf_freq,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// One-pole lowpass filter
// ---------------------------------------------------------------------------

/// One-pole IIR lowpass filter.
///
/// Transfer function: `y[n] = a * x[n] + (1 - a) * y[n-1]`
/// where `a = 1 - exp(-2 * pi * fc / fs)`.
///
/// Unity gain at DC. -3dB at `fc`. -6dB/octave rolloff above.
struct OnePoleLP {
    coeff: f32,
    state: f32,
}

impl OnePoleLP {
    /// Create a new one-pole lowpass for the given cutoff and sample rate.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        // IEEE 754 safety: clamp to prevent NaN/Inf from extreme inputs.
        let fc = cutoff_hz.clamp(1.0, sample_rate * 0.49);
        let coeff = 1.0 - (-2.0 * std::f32::consts::PI * fc / sample_rate).exp();
        let coeff = if !coeff.is_finite() { 1.0 } else { coeff };
        Self { coeff, state: 0.0 }
    }

    /// Process one sample through the filter.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.state = 0.0;
            return 0.0;
        }
        self.state += self.coeff * (x - self.state);
        if !self.state.is_finite() {
            self.state = 0.0;
        }
        self.state
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.state = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Voice bleed (crosstalk)
// ---------------------------------------------------------------------------

/// Apply voice bleed (microphone crosstalk) to a set of voice audio buffers.
///
/// For each voice pair (i, j), a small fraction of voice j's signal is
/// lowpass-filtered and added to voice i. The amount depends on the distance
/// between voices and the configured proximity rolloff.
///
/// # Arguments
///
/// * `voices` - Mutable slice of per-voice PCM buffers (all same length).
/// * `config` - Bleed configuration (amount, rolloff, lowpass cutoff).
/// * `sample_rate` - Audio sample rate in Hz (typically 24000 for Kokoro).
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if config validation fails.
/// Returns `KokoroError::InvalidInput` if voices have mismatched lengths.
pub fn apply_voice_bleed(
    voices: &mut [Vec<f32>],
    config: &BleedConfig,
    sample_rate: u32,
) -> Result<(), KokoroError> {
    config.validate()?;

    let n_voices = voices.len();
    if n_voices <= 1 {
        return Ok(());
    }

    // Bleed amount of 0 is a no-op.
    if config.bleed_amount < 1e-7 {
        return Ok(());
    }

    // Verify all voices have the same length.
    let expected_len = voices[0].len();
    for (i, v) in voices.iter().enumerate().skip(1) {
        if v.len() != expected_len {
            return Err(KokoroError::InvalidInput(format!(
                "voice {} has length {}, expected {} (same as voice 0)",
                i,
                v.len(),
                expected_len,
            )));
        }
    }

    if expected_len == 0 {
        return Ok(());
    }

    let sr = sample_rate as f32;

    // Snapshot all voice data before modification so bleed is computed from
    // original signals, not partially-modified ones.
    let originals: Vec<Vec<f32>> = voices.to_vec();

    for i in 0..n_voices {
        // Accumulate bleed from all other voices into voice i.
        for j in 0..n_voices {
            if i == j {
                continue;
            }

            // Normalized distance in [0, 1].
            let distance = (i as f32 - j as f32).abs() / n_voices as f32;

            // Bleed amount with proximity rolloff.
            let amount = config.bleed_amount / (1.0 + config.proximity_rolloff * distance);

            if amount < 1e-7 {
                continue;
            }

            // Per-pair lowpass filter with deterministic seed-based variation.
            let mut rng = Lcg::new(i.wrapping_mul(31) + j, config.seed);
            let freq_variation = 1.0 + (rng.next_f32() - 0.5) * 0.2; // +/-10%
            let cutoff = (config.lowpass_freq * freq_variation).clamp(100.0, sr * 0.49);
            let mut lp = OnePoleLP::new(cutoff, sr);

            // Filter the source voice signal and add to target voice.
            for k in 0..expected_len {
                let bleed_sample = lp.process(originals[j][k]);
                voices[i][k] += amount * bleed_sample;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Proximity effect (bass boost)
// ---------------------------------------------------------------------------

/// Apply proximity-effect bass boost to a set of voice audio buffers.
///
/// Voice[0] gets the full configured boost. Intermediate voices get
/// linearly decreasing boost. Voice[n-1] gets no boost.
///
/// The boost is implemented as a low-shelf filter: a one-pole lowpass
/// extracts the bass component, which is then scaled and added back.
///
/// # Arguments
///
/// * `voices` - Mutable slice of per-voice PCM buffers.
/// * `config` - Proximity effect configuration.
/// * `sample_rate` - Audio sample rate in Hz.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if config validation fails.
pub fn apply_proximity_effect(
    voices: &mut [Vec<f32>],
    config: &ProximityEffect,
    sample_rate: u32,
) -> Result<(), KokoroError> {
    config.validate()?;

    let n_voices = voices.len();
    if n_voices <= 1 {
        return Ok(());
    }

    // 0 dB boost is a no-op.
    if config.proximity_boost_db < 1e-4 {
        return Ok(());
    }

    let sr = sample_rate as f32;
    let max_linear_boost = db_to_linear(config.proximity_boost_db);

    for (i, voice) in voices.iter_mut().enumerate() {
        // Linear rolloff: voice 0 = full boost, voice n-1 = no boost.
        let voice_fraction = if n_voices > 1 {
            1.0 - (i as f32 / (n_voices - 1) as f32)
        } else {
            1.0
        };

        let boost_linear = 1.0 + (max_linear_boost - 1.0) * voice_fraction;
        if (boost_linear - 1.0).abs() < 1e-6 {
            continue; // No boost for this voice.
        }

        // Extract bass component with a one-pole lowpass, then add scaled
        // version back to create a shelving boost.
        let mut lp = OnePoleLP::new(config.shelf_freq, sr);
        let bass_gain = boost_linear - 1.0; // Additional gain for bass

        for sample in voice.iter_mut() {
            let bass = lp.process(*sample);
            *sample += bass_gain * bass;
        }
    }

    Ok(())
}

/// Convert decibels to linear amplitude (local helper, avoids depending on
/// the saturation module for this simple calculation).
#[inline]
fn db_to_linear(db: f32) -> f32 {
    if !db.is_finite() {
        return 0.0;
    }
    let lin = 10.0_f32.powf(db / 20.0);
    if !lin.is_finite() {
        0.0
    } else {
        lin
    }
}

// ---------------------------------------------------------------------------
// Combined bleed + proximity
// ---------------------------------------------------------------------------

/// Apply both voice bleed and proximity effect in a single call.
///
/// Convenience function that chains [`apply_voice_bleed`] followed by
/// [`apply_proximity_effect`]. Bleed is applied first so the proximity
/// bass boost is applied to the bleed-mixed signal, which is the physical
/// order (crosstalk happens before the proximity effect of the mic).
///
/// # Errors
///
/// Propagates errors from either stage.
pub fn apply_bleed_and_proximity(
    voices: &mut [Vec<f32>],
    bleed_config: &BleedConfig,
    proximity_config: &ProximityEffect,
    sample_rate: u32,
) -> Result<(), KokoroError> {
    apply_voice_bleed(voices, bleed_config, sample_rate)?;
    apply_proximity_effect(voices, proximity_config, sample_rate)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 24000;

    /// Generate a sine wave of the given frequency and length.
    fn sine(freq: f32, len: usize, sr: u32) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    // -- BleedConfig tests --------------------------------------------------

    #[test]
    fn test_bleed_config_defaults_valid() {
        let config = BleedConfig::new();
        config.validate().expect("defaults should be valid");
    }

    #[test]
    fn test_bleed_config_builder() {
        let config = BleedConfig::new()
            .with_bleed_amount(0.10)
            .with_proximity_rolloff(3.0)
            .with_lowpass_freq(1500.0)
            .with_seed(42);
        config.validate().expect("builder config should be valid");
        assert!((config.bleed_amount - 0.10).abs() < 1e-6);
        assert!((config.proximity_rolloff - 3.0).abs() < 1e-6);
        assert!((config.lowpass_freq - 1500.0).abs() < 1e-6);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_bleed_config_rejects_nan() {
        let config = BleedConfig::new().with_bleed_amount(f32::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_bleed_config_rejects_out_of_range() {
        assert!(BleedConfig::new()
            .with_bleed_amount(-0.01)
            .validate()
            .is_err());
        assert!(BleedConfig::new()
            .with_bleed_amount(0.20)
            .validate()
            .is_err());
        assert!(BleedConfig::new()
            .with_proximity_rolloff(0.1)
            .validate()
            .is_err());
        assert!(BleedConfig::new()
            .with_proximity_rolloff(5.0)
            .validate()
            .is_err());
        assert!(BleedConfig::new()
            .with_lowpass_freq(500.0)
            .validate()
            .is_err());
        assert!(BleedConfig::new()
            .with_lowpass_freq(5000.0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_bleed_config_accepts_boundaries() {
        assert!(BleedConfig::new().with_bleed_amount(0.0).validate().is_ok());
        assert!(BleedConfig::new()
            .with_bleed_amount(0.15)
            .validate()
            .is_ok());
        assert!(BleedConfig::new()
            .with_proximity_rolloff(0.5)
            .validate()
            .is_ok());
        assert!(BleedConfig::new()
            .with_proximity_rolloff(4.0)
            .validate()
            .is_ok());
        assert!(BleedConfig::new()
            .with_lowpass_freq(800.0)
            .validate()
            .is_ok());
        assert!(BleedConfig::new()
            .with_lowpass_freq(4000.0)
            .validate()
            .is_ok());
    }

    // -- ProximityEffect tests ----------------------------------------------

    #[test]
    fn test_proximity_defaults_valid() {
        let config = ProximityEffect::new();
        config.validate().expect("defaults should be valid");
    }

    #[test]
    fn test_proximity_rejects_out_of_range() {
        assert!(ProximityEffect::new()
            .with_boost_db(-0.1)
            .validate()
            .is_err());
        assert!(ProximityEffect::new()
            .with_boost_db(7.0)
            .validate()
            .is_err());
        assert!(ProximityEffect::new()
            .with_shelf_freq(50.0)
            .validate()
            .is_err());
        assert!(ProximityEffect::new()
            .with_shelf_freq(600.0)
            .validate()
            .is_err());
    }

    // -- apply_voice_bleed tests --------------------------------------------

    #[test]
    fn test_bleed_zero_amount_is_identity() {
        let config = BleedConfig::new().with_bleed_amount(0.0);
        let original = sine(440.0, 2000, SR);
        let mut voices = vec![original.clone(), original.clone()];
        apply_voice_bleed(&mut voices, &config, SR).unwrap();

        for (i, voice) in voices.iter().enumerate() {
            for (j, (&got, &expected)) in voice.iter().zip(original.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-7,
                    "voice {i} sample {j}: bleed_amount=0 should be identity"
                );
            }
        }
    }

    #[test]
    fn test_bleed_single_voice_unchanged() {
        let config = BleedConfig::new().with_bleed_amount(0.10);
        let original = sine(440.0, 2000, SR);
        let mut voices = vec![original.clone()];
        apply_voice_bleed(&mut voices, &config, SR).unwrap();

        for (j, (&got, &expected)) in voices[0].iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-7,
                "single voice sample {j}: should be unchanged"
            );
        }
    }

    #[test]
    fn test_bleed_two_voices_adds_crosstalk() {
        let config = BleedConfig::new()
            .with_bleed_amount(0.10)
            .with_proximity_rolloff(1.0)
            .with_lowpass_freq(4000.0); // high cutoff so most signal passes

        let voice_a = sine(440.0, 4000, SR);
        let voice_b: Vec<f32> = vec![0.0; 4000]; // silence
        let mut voices = vec![voice_a.clone(), voice_b];
        apply_voice_bleed(&mut voices, &config, SR).unwrap();

        // Voice A should be mostly unchanged (only bleed from silence).
        let diff_a: f32 = voices[0]
            .iter()
            .zip(voice_a.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / voices[0].len() as f32;
        assert!(
            diff_a < 1e-5,
            "voice A should barely change when B is silent, mean diff = {diff_a}"
        );

        // Voice B (was silent) should now contain filtered bleed from A.
        let rms_b: f32 =
            (voices[1].iter().map(|x| x * x).sum::<f32>() / voices[1].len() as f32).sqrt();
        assert!(
            rms_b > 1e-4,
            "voice B should have nonzero bleed from A, RMS = {rms_b}"
        );
    }

    #[test]
    fn test_bleed_does_not_exceed_15_percent() {
        // Maximum configured bleed is 0.15. For adjacent voices (distance ~0.25
        // with 4 voices, rolloff=0.5), the actual bleed per pair is:
        //   0.15 / (1 + 0.5 * 0.25) = 0.15 / 1.125 ≈ 0.133
        // Total bleed into one voice from 3 neighbors is at most ~3 * 0.133.
        // But each individual pair contribution should be < 15% of source RMS.
        let config = BleedConfig::new()
            .with_bleed_amount(0.15)
            .with_proximity_rolloff(0.5)
            .with_lowpass_freq(4000.0);

        let signal = sine(440.0, 4000, SR);
        let rms_original: f32 =
            (signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32).sqrt();

        let silence = vec![0.0f32; 4000];
        let mut voices = vec![
            signal,
            silence.clone(),
            silence.clone(),
            silence,
        ];
        apply_voice_bleed(&mut voices, &config, SR).unwrap();

        // Each silent voice should have bleed < 15% of original RMS.
        for (i, voice) in voices.iter().enumerate().skip(1) {
            let rms: f32 = (voice.iter().map(|x| x * x).sum::<f32>() / voice.len() as f32).sqrt();
            let ratio = rms / rms_original;
            assert!(
                ratio < 0.16,
                "voice {i} bleed ratio {ratio:.4} should be < 0.16"
            );
        }
    }

    #[test]
    fn test_bleed_four_voices() {
        let config = BleedConfig::new()
            .with_bleed_amount(0.05)
            .with_proximity_rolloff(2.0);

        let mut voices: Vec<Vec<f32>> = (0..4)
            .map(|i| sine(220.0 * (i as f32 + 1.0), 2000, SR))
            .collect();
        let originals: Vec<Vec<f32>> = voices.clone();

        apply_voice_bleed(&mut voices, &config, SR).unwrap();

        // All voices should be slightly modified.
        for (i, (voice, orig)) in voices.iter().zip(originals.iter()).enumerate() {
            let diff: f32 = voice
                .iter()
                .zip(orig.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / voice.len() as f32;
            assert!(
                diff > 1e-5,
                "voice {i} should be modified by bleed, mean diff = {diff}"
            );
        }
    }

    #[test]
    fn test_bleed_eight_voices() {
        let config = BleedConfig::new()
            .with_bleed_amount(0.08)
            .with_proximity_rolloff(3.0);

        let mut voices: Vec<Vec<f32>> = (0..8)
            .map(|i| sine(200.0 + 50.0 * i as f32, 2000, SR))
            .collect();
        apply_voice_bleed(&mut voices, &config, SR).unwrap();

        // Verify all voices are finite.
        for (i, voice) in voices.iter().enumerate() {
            for (j, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "voice {i} sample {j} is not finite: {s}");
            }
        }
    }

    #[test]
    fn test_bleed_lowpass_attenuates_highs() {
        // Use a very low cutoff to strongly filter the bleed signal.
        let config = BleedConfig::new()
            .with_bleed_amount(0.10)
            .with_proximity_rolloff(0.5)
            .with_lowpass_freq(800.0);

        // Source is a high-frequency sine (8kHz) — should be heavily attenuated.
        let hf_signal = sine(8000.0, 4000, SR);
        let silence = vec![0.0f32; 4000];
        let mut voices = vec![hf_signal.clone(), silence];
        apply_voice_bleed(&mut voices, &config, SR).unwrap();

        // RMS of bleed into voice 1 should be very small (heavy LP attenuation).
        let rms_bleed: f32 =
            (voices[1].iter().map(|x| x * x).sum::<f32>() / voices[1].len() as f32).sqrt();
        let rms_source: f32 =
            (hf_signal.iter().map(|x| x * x).sum::<f32>() / hf_signal.len() as f32).sqrt();

        let ratio = rms_bleed / rms_source;
        assert!(
            ratio < 0.05,
            "8kHz signal through 800Hz LP should be heavily attenuated, ratio = {ratio:.4}"
        );
    }

    #[test]
    fn test_bleed_mismatched_lengths_error() {
        let config = BleedConfig::new();
        let mut voices = vec![vec![0.0; 100], vec![0.0; 200]];
        assert!(apply_voice_bleed(&mut voices, &config, SR).is_err());
    }

    #[test]
    fn test_bleed_empty_voices_ok() {
        let config = BleedConfig::new();
        let mut voices: Vec<Vec<f32>> = vec![];
        assert!(apply_voice_bleed(&mut voices, &config, SR).is_ok());
    }

    #[test]
    fn test_bleed_empty_buffers_ok() {
        let config = BleedConfig::new();
        let mut voices: Vec<Vec<f32>> = vec![vec![], vec![]];
        assert!(apply_voice_bleed(&mut voices, &config, SR).is_ok());
    }

    // -- apply_proximity_effect tests ---------------------------------------

    #[test]
    fn test_proximity_zero_boost_is_identity() {
        let config = ProximityEffect::new().with_boost_db(0.0);
        let original = sine(440.0, 2000, SR);
        let mut voices = vec![original.clone(), original.clone()];
        apply_proximity_effect(&mut voices, &config, SR).unwrap();

        for (i, voice) in voices.iter().enumerate() {
            for (j, (&got, &expected)) in voice.iter().zip(original.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "voice {i} sample {j}: 0dB boost should be identity"
                );
            }
        }
    }

    #[test]
    fn test_proximity_voice0_gets_most_boost() {
        let config = ProximityEffect::new().with_boost_db(6.0);
        let signal = sine(100.0, 4000, SR); // Low frequency to be boosted
        let mut voices = vec![signal.clone(), signal.clone(), signal];
        let originals: Vec<Vec<f32>> = voices.clone();

        apply_proximity_effect(&mut voices, &config, SR).unwrap();

        // Measure RMS change per voice.
        let rms_changes: Vec<f32> = voices
            .iter()
            .zip(originals.iter())
            .map(|(v, o)| {
                let diff: f32 = v
                    .iter()
                    .zip(o.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();
                (diff / v.len() as f32).sqrt()
            })
            .collect();

        // Voice 0 should have the most change, voice 2 the least.
        assert!(
            rms_changes[0] > rms_changes[1],
            "voice 0 change ({:.6}) should exceed voice 1 ({:.6})",
            rms_changes[0],
            rms_changes[1],
        );
        assert!(
            rms_changes[1] > rms_changes[2],
            "voice 1 change ({:.6}) should exceed voice 2 ({:.6})",
            rms_changes[1],
            rms_changes[2],
        );
    }

    #[test]
    fn test_proximity_last_voice_unchanged() {
        let config = ProximityEffect::new().with_boost_db(4.0);
        let signal = sine(100.0, 2000, SR);
        let mut voices = vec![signal.clone(), signal.clone(), signal.clone()];
        apply_proximity_effect(&mut voices, &config, SR).unwrap();

        // Last voice should be unmodified (0 boost fraction).
        for (j, (&got, &expected)) in voices[2].iter().zip(signal.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "last voice sample {j}: should be unchanged, got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_proximity_single_voice_ok() {
        let config = ProximityEffect::new().with_boost_db(3.0);
        let signal = sine(200.0, 1000, SR);
        let mut voices = vec![signal.clone()];
        apply_proximity_effect(&mut voices, &config, SR).unwrap();
        // Single voice path: n_voices <= 1, early return.
        for (j, (&got, &expected)) in voices[0].iter().zip(signal.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "single voice sample {j}: unchanged"
            );
        }
    }

    // -- Combined bleed + proximity -----------------------------------------

    #[test]
    fn test_combined_bleed_and_proximity() {
        let bleed_config = BleedConfig::new().with_bleed_amount(0.05);
        let proximity_config = ProximityEffect::new().with_boost_db(2.0);

        let mut voices: Vec<Vec<f32>> = (0..4)
            .map(|i| sine(200.0 + 100.0 * i as f32, 2000, SR))
            .collect();

        apply_bleed_and_proximity(&mut voices, &bleed_config, &proximity_config, SR).unwrap();

        // All voices should be finite.
        for (i, voice) in voices.iter().enumerate() {
            for (j, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "combined: voice {i} sample {j} not finite");
            }
        }
    }

    // -- One-pole LP filter test --------------------------------------------

    #[test]
    fn test_one_pole_lp_attenuates_above_cutoff() {
        let mut lp = OnePoleLP::new(1000.0, 24000.0);
        let n = 4096;

        // Low frequency signal (200 Hz) — should pass mostly through.
        let low: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 24000.0).sin())
            .collect();
        let low_out: Vec<f32> = low.iter().map(|&s| lp.process(s)).collect();

        lp.reset();

        // High frequency signal (10 kHz) — should be attenuated.
        let high: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 10000.0 * i as f32 / 24000.0).sin())
            .collect();
        let high_out: Vec<f32> = high.iter().map(|&s| lp.process(s)).collect();

        // Compare RMS after transient (skip 128 samples).
        let skip = 128;
        let rms = |data: &[f32]| -> f32 {
            (data[skip..].iter().map(|x| x * x).sum::<f32>() / (data.len() - skip) as f32).sqrt()
        };

        let low_ratio = rms(&low_out) / rms(&low);
        let high_ratio = rms(&high_out) / rms(&high);

        assert!(
            low_ratio > 0.7,
            "200Hz should pass mostly through 1kHz LP, ratio = {low_ratio:.4}"
        );
        assert!(
            high_ratio < 0.3,
            "10kHz should be attenuated by 1kHz LP, ratio = {high_ratio:.4}"
        );
    }

    #[test]
    fn test_one_pole_lp_nan_safety() {
        let mut lp = OnePoleLP::new(1000.0, 24000.0);
        let out = lp.process(f32::NAN);
        assert!(out.abs() < 1e-6, "NaN input should produce 0.0, got {out}");
        // Filter should recover.
        let out = lp.process(1.0);
        assert!(out.is_finite(), "should recover after NaN");
    }

    // -- Determinism test ---------------------------------------------------

    #[test]
    fn test_bleed_is_deterministic() {
        let config = BleedConfig::new().with_bleed_amount(0.08).with_seed(123);

        let make_voices = || -> Vec<Vec<f32>> {
            (0..3)
                .map(|i| sine(300.0 + 100.0 * i as f32, 2000, SR))
                .collect()
        };

        let mut v1 = make_voices();
        let mut v2 = make_voices();
        apply_voice_bleed(&mut v1, &config, SR).unwrap();
        apply_voice_bleed(&mut v2, &config, SR).unwrap();

        for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
            for (j, (&sa, &sb)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (sa - sb).abs() < 1e-7,
                    "determinism: voice {i} sample {j}: {sa} != {sb}"
                );
            }
        }
    }
}
