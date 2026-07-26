// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Voice onset synchronization for multi-voice Kokoro chorus timing control.
//!
//! Real choirs have slight onset variations -- singers do not begin each phrase
//! at precisely the same instant. This module detects onsets (transient energy
//! rises) in each voice, aligns them to a configurable lead voice, and applies
//! controllable tightness so the ensemble feels anywhere from perfectly locked
//! to loosely staggered.
//!
//! # Algorithm
//!
//! 1. **Detect onsets** in every voice: compute a sliding-window RMS energy
//!    envelope, take its first-order difference, and find positive threshold
//!    crossings.
//! 2. For each onset in the lead voice, find the nearest corresponding onset
//!    in every other voice (within `max_shift_ms`).
//! 3. Compute the time offset (how early or late each voice is relative to the
//!    lead).
//! 4. Apply correction: shift each voice by `tightness * offset` samples using
//!    a fractional-delay line with linear interpolation.
//! 5. Optionally add a per-voice stagger (`i * stagger_ms`) for cascading
//!    width after alignment.
//!
//! Part of #4582.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default sample rate for time conversions (Kokoro outputs 24 kHz).
const DEFAULT_SAMPLE_RATE: u32 = 24_000;

/// Minimum gap between consecutive detected onsets (in samples at 24 kHz).
/// Prevents double-triggers from multi-peak transients.
const MIN_ONSET_GAP_SAMPLES: usize = 480; // 20 ms

/// RMS analysis hop size in samples.
const RMS_HOP: usize = 64;

/// RMS analysis window size in samples.
const RMS_WINDOW: usize = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for voice onset synchronization.
///
/// Controls how tightly onsets are aligned across voices and whether an
/// intentional per-voice stagger is applied after alignment for spatial width.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OnsetSyncConfig {
    /// Alignment tightness: 0.0 = no correction (natural), 1.0 = perfectly tight.
    ///
    /// At intermediate values the correction is proportional: the voice is
    /// shifted by `tightness * measured_offset` samples.
    ///
    /// Range: [0.0, 1.0]. Default: `0.7`.
    pub tightness: f32,

    /// Maximum time adjustment in milliseconds.
    ///
    /// Onsets further apart than this are assumed to be unrelated and not
    /// corrected. Also caps the per-voice delay line length.
    ///
    /// Range: [1.0, 100.0]. Default: `20.0`.
    pub max_shift_ms: f32,

    /// Onset detection threshold in dB below full scale.
    ///
    /// Energy rises above this threshold (in the first-difference of the RMS
    /// envelope) are classified as onsets. More negative = more sensitive.
    ///
    /// Range: [-60.0, 0.0]. Default: `-30.0`.
    pub detection_threshold_db: f32,

    /// Index of the lead voice that defines the timing reference.
    ///
    /// All other voices are shifted to align with this voice's onsets.
    ///
    /// Default: `0`.
    pub lead_voice: usize,

    /// Intentional per-voice stagger in milliseconds, applied after alignment.
    ///
    /// Voice `i` receives an additional delay of `i * stagger_ms`. This creates
    /// a cascading onset pattern that adds spatial width while keeping the
    /// overall ensemble tight.
    ///
    /// Range: [0.0, 20.0]. Default: `0.0` (no stagger).
    pub stagger_ms: f32,

    /// Sample rate for time conversions. Default: `24000`.
    pub sample_rate: u32,
}

impl Default for OnsetSyncConfig {
    fn default() -> Self {
        Self {
            tightness: 0.7,
            max_shift_ms: 20.0,
            detection_threshold_db: -30.0,
            lead_voice: 0,
            stagger_ms: 0.0,
            sample_rate: DEFAULT_SAMPLE_RATE,
        }
    }
}

impl OnsetSyncConfig {
    /// Create a config with the given tightness and defaults for the rest.
    pub fn new(tightness: f32) -> Result<Self, KokoroError> {
        let config = Self {
            tightness,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Tight unison preset: near-perfect synchronization.
    #[must_use]
    pub fn tight_unison() -> Self {
        Self {
            tightness: 0.95,
            ..Self::default()
        }
    }

    /// Natural chorus preset: moderate correction preserving human feel.
    #[must_use]
    pub fn natural_chorus() -> Self {
        Self {
            tightness: 0.6,
            ..Self::default()
        }
    }

    /// Loose ensemble preset: gentle correction, mostly natural timing.
    #[must_use]
    pub fn loose_ensemble() -> Self {
        Self {
            tightness: 0.3,
            max_shift_ms: 30.0,
            ..Self::default()
        }
    }

    /// Staggered preset: moderate tightness with cascading voice delays.
    #[must_use]
    pub fn staggered() -> Self {
        Self {
            tightness: 0.7,
            stagger_ms: 3.0,
            ..Self::default()
        }
    }

    // -- Builder methods ---------------------------------------------------

    /// Set maximum shift in milliseconds.
    #[must_use]
    pub fn with_max_shift_ms(mut self, ms: f32) -> Self {
        self.max_shift_ms = ms;
        self
    }

    /// Set detection threshold in dB.
    #[must_use]
    pub fn with_threshold_db(mut self, db: f32) -> Self {
        self.detection_threshold_db = db;
        self
    }

    /// Set lead voice index.
    #[must_use]
    pub fn with_lead_voice(mut self, idx: usize) -> Self {
        self.lead_voice = idx;
        self
    }

    /// Set per-voice stagger in milliseconds.
    #[must_use]
    pub fn with_stagger_ms(mut self, ms: f32) -> Self {
        self.stagger_ms = ms;
        self
    }

    /// Set sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Validate all fields. Returns `Err` on out-of-range or non-finite values.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.tightness.is_finite() || !(0.0..=1.0).contains(&self.tightness) {
            return Err(KokoroError::InvalidConfig {
                field: "tightness",
                reason: format!("must be finite and in [0.0, 1.0], got {}", self.tightness),
            });
        }
        if !self.max_shift_ms.is_finite() || !(1.0..=100.0).contains(&self.max_shift_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "max_shift_ms",
                reason: format!("must be in [1.0, 100.0], got {}", self.max_shift_ms),
            });
        }
        if !self.detection_threshold_db.is_finite()
            || !(-60.0..=0.0).contains(&self.detection_threshold_db)
        {
            return Err(KokoroError::InvalidConfig {
                field: "detection_threshold_db",
                reason: format!(
                    "must be in [-60.0, 0.0], got {}",
                    self.detection_threshold_db
                ),
            });
        }
        if !self.stagger_ms.is_finite() || !(0.0..=20.0).contains(&self.stagger_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "stagger_ms",
                reason: format!("must be in [0.0, 20.0], got {}", self.stagger_ms),
            });
        }
        if self.sample_rate == 0 || self.sample_rate > 192_000 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be in [1, 192000], got {}", self.sample_rate),
            });
        }
        Ok(())
    }

    /// Maximum shift expressed in samples at the configured sample rate.
    fn max_shift_samples(&self) -> usize {
        ((self.max_shift_ms / 1000.0) * self.sample_rate as f32).ceil() as usize
    }

    /// Per-voice stagger expressed in samples.
    fn stagger_samples(&self) -> f32 {
        (self.stagger_ms / 1000.0) * self.sample_rate as f32
    }

    /// Linear amplitude corresponding to `detection_threshold_db`.
    #[must_use]
    pub fn threshold_linear(&self) -> f32 {
        10.0_f32.powf(self.detection_threshold_db / 20.0)
    }
}

// ---------------------------------------------------------------------------
// Onset detection
// ---------------------------------------------------------------------------

/// Detect onset positions in an audio signal.
///
/// Uses a sliding-window RMS energy envelope and its first-order difference.
/// Positive threshold crossings with sufficient inter-onset spacing are
/// returned as sample indices.
///
/// # Arguments
///
/// * `audio` - Mono audio samples (f32, any range).
/// * `threshold_db` - Detection threshold in dB below full scale.
/// * `sample_rate` - For computing minimum onset gap.
///
/// Returns sorted onset sample indices.
pub fn detect_onsets(audio: &[f32], threshold_db: f32, sample_rate: u32) -> Vec<usize> {
    if audio.len() < RMS_WINDOW {
        return Vec::new();
    }

    let threshold_linear = 10.0_f32.powf(threshold_db / 20.0);
    let min_gap = MIN_ONSET_GAP_SAMPLES.max(
        ((sample_rate as f32 / DEFAULT_SAMPLE_RATE as f32) * MIN_ONSET_GAP_SAMPLES as f32) as usize,
    );

    // Compute RMS envelope at hop intervals.
    let num_frames = (audio.len().saturating_sub(RMS_WINDOW)) / RMS_HOP + 1;
    let mut rms_env = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * RMS_HOP;
        let end = (start + RMS_WINDOW).min(audio.len());
        let sum_sq: f32 = audio[start..end].iter().map(|&s| s * s).sum();
        let count = (end - start) as f32;
        rms_env.push((sum_sq / count).sqrt());
    }

    // First-order difference of RMS envelope (energy rise detector).
    let mut onsets = Vec::new();
    let mut last_onset_sample: Option<usize> = None;

    for i in 1..rms_env.len() {
        let diff = rms_env[i] - rms_env[i - 1];
        if diff > threshold_linear {
            let sample_pos = i * RMS_HOP;
            if let Some(last) = last_onset_sample {
                if sample_pos.saturating_sub(last) < min_gap {
                    continue;
                }
            }
            onsets.push(sample_pos);
            last_onset_sample = Some(sample_pos);
        }
    }

    onsets
}

// ---------------------------------------------------------------------------
// Fractional delay line
// ---------------------------------------------------------------------------

/// Apply a fractional sample delay to an audio buffer using linear interpolation.
///
/// Positive `delay` shifts the signal later (inserts silence at the start).
/// Negative `delay` shifts the signal earlier (trims from the start).
/// Fractional parts are handled via linear interpolation between adjacent samples.
fn apply_fractional_delay(audio: &[f32], delay: f32) -> Vec<f32> {
    let len = audio.len();
    if len == 0 {
        return Vec::new();
    }

    let mut output = vec![0.0f32; len];

    for i in 0..len {
        // Source position: where in the original signal does output[i] come from?
        let src = i as f32 - delay;
        if src < 0.0 || src >= (len - 1) as f32 {
            // Outside bounds: silence (zero).
            continue;
        }
        let idx = src.floor() as usize;
        let frac = src - src.floor();
        // Linear interpolation.
        let s0 = audio[idx];
        let s1 = if idx + 1 < len { audio[idx + 1] } else { s0 };
        output[i] = s0 + frac * (s1 - s0);
    }

    output
}

// ---------------------------------------------------------------------------
// Onset synchronizer
// ---------------------------------------------------------------------------

/// Stateful voice onset synchronizer.
///
/// Detects onsets across multiple voices, aligns them to the lead voice,
/// and applies optional per-voice stagger for spatial width.
#[derive(Debug, Clone)]
pub struct OnsetSynchronizer {
    config: OnsetSyncConfig,
    /// Cached onset positions per voice from the last `synchronize` call.
    cached_onsets: Vec<Vec<usize>>,
}

impl OnsetSynchronizer {
    /// Create a new synchronizer with the given config.
    pub fn new(config: OnsetSyncConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        Ok(Self {
            config,
            cached_onsets: Vec::new(),
        })
    }

    /// Create a synchronizer with default config.
    pub fn with_defaults() -> Self {
        Self {
            config: OnsetSyncConfig::default(),
            cached_onsets: Vec::new(),
        }
    }

    /// Access the current config.
    #[must_use]
    pub fn config(&self) -> &OnsetSyncConfig {
        &self.config
    }

    /// Access cached onsets from the last `synchronize` call.
    #[must_use]
    pub fn cached_onsets(&self) -> &[Vec<usize>] {
        &self.cached_onsets
    }

    /// Reset internal state (cached onsets).
    pub fn reset(&mut self) {
        self.cached_onsets.clear();
    }

    /// Synchronize onsets across all voices in-place.
    ///
    /// Each voice is an `&mut Vec<f32>` of mono audio. All voices must have the
    /// same length. The lead voice (by index) defines the timing reference;
    /// other voices are shifted to match its onsets.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `voices` is empty, lengths differ, or the lead voice
    /// index is out of range.
    pub fn synchronize(&mut self, voices: &mut [Vec<f32>]) -> Result<(), KokoroError> {
        if voices.is_empty() {
            return Err(KokoroError::InvalidConfig {
                field: "voices",
                reason: "at least one voice required".to_string(),
            });
        }
        if self.config.lead_voice >= voices.len() {
            return Err(KokoroError::InvalidConfig {
                field: "lead_voice",
                reason: format!(
                    "lead_voice {} >= voice count {}",
                    self.config.lead_voice,
                    voices.len()
                ),
            });
        }
        let expected_len = voices[0].len();
        for (i, v) in voices.iter().enumerate() {
            if v.len() != expected_len {
                return Err(KokoroError::InvalidConfig {
                    field: "voices",
                    reason: format!(
                        "voice {} has length {}, expected {}",
                        i,
                        v.len(),
                        expected_len
                    ),
                });
            }
        }

        // Passthrough if tightness is zero and no stagger.
        if self.config.tightness == 0.0 && self.config.stagger_ms == 0.0 {
            self.cached_onsets = voices
                .iter()
                .map(|v| {
                    detect_onsets(
                        v,
                        self.config.detection_threshold_db,
                        self.config.sample_rate,
                    )
                })
                .collect();
            return Ok(());
        }

        // Step 1: Detect onsets in every voice.
        let all_onsets: Vec<Vec<usize>> = voices
            .iter()
            .map(|v| {
                detect_onsets(
                    v,
                    self.config.detection_threshold_db,
                    self.config.sample_rate,
                )
            })
            .collect();

        let lead_onsets = &all_onsets[self.config.lead_voice];
        let max_shift = self.config.max_shift_samples();
        let stagger_per_voice = self.config.stagger_samples();

        // Step 2-5: For each non-lead voice, compute aggregate shift and apply.
        for (voice_idx, voice) in voices.iter_mut().enumerate() {
            if voice_idx == self.config.lead_voice {
                continue;
            }

            let voice_onsets = &all_onsets[voice_idx];

            // Compute average offset across matched onset pairs.
            let avg_offset = compute_average_offset(lead_onsets, voice_onsets, max_shift);

            // Correction: shift the voice closer to the lead by tightness fraction.
            let correction = self.config.tightness * avg_offset;

            // Add intentional stagger.
            let voice_distance = voice_idx.abs_diff(self.config.lead_voice);
            let stagger_delay = voice_distance as f32 * stagger_per_voice;

            let total_delay = correction + stagger_delay;

            // Apply the shift.
            if total_delay.abs() > 0.01 {
                let shifted = apply_fractional_delay(voice, total_delay);
                voice.copy_from_slice(&shifted);
            }
        }

        self.cached_onsets = all_onsets;
        Ok(())
    }
}

/// Compute the average onset time offset between a lead and a follower voice.
///
/// For each lead onset, find the nearest follower onset within `max_shift`
/// samples and accumulate the signed offset. Returns the mean offset (positive
/// means the follower is late, negative means early). If no pairs are matched,
/// returns 0.0.
fn compute_average_offset(
    lead_onsets: &[usize],
    follower_onsets: &[usize],
    max_shift: usize,
) -> f32 {
    if lead_onsets.is_empty() || follower_onsets.is_empty() {
        return 0.0;
    }

    let mut total_offset: f64 = 0.0;
    let mut count: u32 = 0;
    let mut search_start = 0;

    for &lead_pos in lead_onsets {
        // Binary-search-like scan for closest follower onset.
        let mut best_dist = max_shift as i64 + 1;
        let mut best_offset: i64 = 0;

        for j in search_start..follower_onsets.len() {
            let f_pos = follower_onsets[j] as i64;
            let l_pos = lead_pos as i64;
            let dist = (f_pos - l_pos).abs();

            if dist < best_dist {
                best_dist = dist;
                best_offset = f_pos - l_pos;
            }

            // Once we're past the lead onset by more than max_shift, we can
            // stop scanning for this lead onset. Update search_start for the
            // next iteration.
            if f_pos > l_pos + max_shift as i64 {
                break;
            }
        }

        if best_dist <= max_shift as i64 {
            // Offset is positive when the follower is late relative to lead.
            total_offset += best_offset as f64;
            count += 1;
            // Advance search start for monotonicity.
            if search_start < follower_onsets.len() {
                search_start = search_start.saturating_sub(1);
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    (total_offset / f64::from(count)) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onset_sync_config_default_valid() {
        OnsetSyncConfig::default()
            .validate()
            .expect("default should be valid");
    }

    #[test]
    fn test_onset_sync_config_presets_valid() {
        OnsetSyncConfig::tight_unison()
            .validate()
            .expect("tight_unison valid");
        OnsetSyncConfig::natural_chorus()
            .validate()
            .expect("natural_chorus valid");
        OnsetSyncConfig::loose_ensemble()
            .validate()
            .expect("loose_ensemble valid");
        OnsetSyncConfig::staggered()
            .validate()
            .expect("staggered valid");
    }

    #[test]
    fn test_onset_sync_config_invalid_tightness() {
        let config = OnsetSyncConfig {
            tightness: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = OnsetSyncConfig {
            tightness: -0.1,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = OnsetSyncConfig {
            tightness: f32::NAN,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_onset_sync_config_invalid_max_shift() {
        let config = OnsetSyncConfig {
            max_shift_ms: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = OnsetSyncConfig {
            max_shift_ms: 200.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_onset_sync_config_builder() {
        let config = OnsetSyncConfig::default()
            .with_max_shift_ms(10.0)
            .with_threshold_db(-20.0)
            .with_lead_voice(1)
            .with_stagger_ms(2.0)
            .with_sample_rate(48_000);
        config.validate().expect("builder config valid");
        assert_eq!(config.lead_voice, 1);
        assert!((config.stagger_ms - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_onsets_silence() {
        let silence = vec![0.0f32; 4800];
        let onsets = detect_onsets(&silence, -30.0, 24_000);
        assert!(onsets.is_empty(), "silence should have no onsets");
    }

    #[test]
    fn test_detect_onsets_impulse() {
        // Create a signal with a clear onset: silence then a burst.
        let mut signal = vec![0.0f32; 4800];
        for s in &mut signal[2400..2800] {
            *s = 0.8;
        }
        let onsets = detect_onsets(&signal, -30.0, 24_000);
        assert!(!onsets.is_empty(), "should detect the impulse onset");
        // The onset should be near sample 2400.
        let first = onsets[0];
        assert!(
            (2200..2700).contains(&first),
            "onset at {first} should be near 2400"
        );
    }

    #[test]
    fn test_detect_onsets_short_signal() {
        let short = vec![0.5f32; 100];
        let onsets = detect_onsets(&short, -30.0, 24_000);
        // Signal shorter than RMS_WINDOW => no onsets.
        assert!(onsets.is_empty());
    }

    #[test]
    fn test_fractional_delay_zero() {
        let signal: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let out = apply_fractional_delay(&signal, 0.0);
        assert_eq!(out.len(), signal.len());
        for (a, b) in signal.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fractional_delay_integer() {
        let signal: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = apply_fractional_delay(&signal, 5.0);
        // First 5 samples should be zero (silence inserted).
        for &s in &out[0..5] {
            assert!((s).abs() < 1e-6, "expected silence, got {s}");
        }
        // Sample at index 5 should be approximately signal[0] = 0.0.
        assert!((out[5] - 0.0).abs() < 1e-4);
        // Sample at index 10 should be approximately signal[5] = 5.0.
        assert!((out[10] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_synchronize_single_voice() {
        let mut sync = OnsetSynchronizer::with_defaults();
        let mut voices = vec![vec![0.5f32; 4800]];
        sync.synchronize(&mut voices)
            .expect("single voice should succeed");
    }

    #[test]
    fn test_synchronize_empty_voices_err() {
        let mut sync = OnsetSynchronizer::with_defaults();
        let mut voices: Vec<Vec<f32>> = Vec::new();
        assert!(sync.synchronize(&mut voices).is_err());
    }

    #[test]
    fn test_synchronize_lead_out_of_range() {
        let config = OnsetSyncConfig::default().with_lead_voice(5);
        let mut sync = OnsetSynchronizer::new(config).expect("config valid");
        let mut voices = vec![vec![0.0f32; 4800]; 2];
        assert!(sync.synchronize(&mut voices).is_err());
    }

    #[test]
    fn test_synchronize_mismatched_lengths() {
        let mut sync = OnsetSynchronizer::with_defaults();
        let mut voices = vec![vec![0.0f32; 4800], vec![0.0f32; 2400]];
        assert!(sync.synchronize(&mut voices).is_err());
    }

    #[test]
    fn test_synchronize_zero_tightness_passthrough() {
        let config = OnsetSyncConfig {
            tightness: 0.0,
            stagger_ms: 0.0,
            ..Default::default()
        };
        let mut sync = OnsetSynchronizer::new(config).expect("config valid");
        let original = vec![0.3f32; 4800];
        let mut voices = vec![original.clone(), original];
        sync.synchronize(&mut voices).expect("should pass through");
        // Voices should be unchanged.
        assert_eq!(voices[0], voices[1]);
    }

    #[test]
    fn test_synchronize_preserves_lead_voice() {
        let mut sync = OnsetSynchronizer::with_defaults();
        // Lead voice is 0 by default.
        let lead = vec![0.5f32; 4800];
        let follower = vec![0.3f32; 4800];
        let lead_copy = lead.clone();
        let mut voices = vec![lead, follower];
        sync.synchronize(&mut voices).expect("sync ok");
        // Lead voice must be untouched.
        assert_eq!(voices[0], lead_copy);
    }

    #[test]
    fn test_compute_average_offset_no_onsets() {
        assert!((compute_average_offset(&[], &[100, 200], 480)).abs() < f32::EPSILON);
        assert!((compute_average_offset(&[100], &[], 480)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_average_offset_aligned() {
        let lead = vec![1000, 2000, 3000];
        let follower = vec![1000, 2000, 3000];
        let offset = compute_average_offset(&lead, &follower, 480);
        assert!(
            offset.abs() < 1.0,
            "perfectly aligned should give ~0 offset"
        );
    }

    #[test]
    fn test_compute_average_offset_late_follower() {
        let lead = vec![1000, 2000];
        let follower = vec![1100, 2100]; // 100 samples late
        let offset = compute_average_offset(&lead, &follower, 480);
        assert!(
            (offset - 100.0).abs() < 1.0,
            "expected ~100 offset, got {offset}"
        );
    }

    #[test]
    fn test_onset_synchronizer_reset() {
        let mut sync = OnsetSynchronizer::with_defaults();
        let mut voices = vec![vec![0.0f32; 4800]; 2];
        sync.synchronize(&mut voices).expect("ok");
        assert!(!sync.cached_onsets().is_empty());
        sync.reset();
        assert!(sync.cached_onsets().is_empty());
    }
}
