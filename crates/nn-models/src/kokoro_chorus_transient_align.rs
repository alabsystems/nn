// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Transient attack alignment across multi-voice chorus.
//!
//! In a multi-voice TTS chorus, each voice produces slightly different onset
//! timing for consonants and plosive bursts. When these onsets are misaligned
//! by more than ~2-3 ms the result sounds sloppy rather than lush. Real
//! choirs achieve tight onset timing on stressed syllables.
//!
//! This module detects transient onsets in each voice using energy-derivative
//! analysis and time-aligns them to a common reference (earliest or average
//! onset) by micro-shifting the attack portion while preserving the natural
//! sustain and release envelope.
//!
//! # Architecture
//!
//! ```text
//! Voice[i] --> circular look-ahead buffer --> energy derivative
//!          --> onset detection (threshold crossing)
//!          --> compute target time (earliest / average across voices)
//!          --> micro-shift attack window to target
//!          --> crossfade with original sustain/release
//!          --> Aligned Voice[i]
//! ```
//!
//! Part of #4264, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for transient attack alignment across chorus voices.
///
/// Controls onset detection sensitivity, alignment tightness, and the
/// look-ahead window used for non-causal onset detection.
///
/// Constructed via [`TransientAlignConfig::new`] or presets (required for
/// cross-crate use due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TransientAlignConfig {
    /// Detection threshold in dB below peak energy derivative.
    /// Onsets are detected when the energy derivative exceeds this threshold.
    /// Range: [-60.0, -6.0]. Default: -30.0.
    pub detection_threshold_db: f32,
    /// Alignment strength: 0.0 = no alignment, 1.0 = snap to target time.
    /// Intermediate values interpolate between original and aligned onset.
    /// Range: [0.0, 1.0]. Default: 0.5.
    pub alignment_strength: f32,
    /// Look-ahead in milliseconds for non-causal onset detection.
    /// A small look-ahead lets the detector see the rising edge before it
    /// arrives, improving detection accuracy.
    /// Range: [0.5, 20.0]. Default: 5.0.
    pub lookahead_ms: f32,
    /// Attack window in milliseconds: the region around each detected onset
    /// that is eligible for micro-shifting. Only this window is moved;
    /// sustain and release are preserved.
    /// Range: [1.0, 50.0]. Default: 10.0.
    pub attack_window_ms: f32,
    /// Maximum shift in milliseconds: the largest onset correction allowed.
    /// Larger values can fix bigger timing errors but risk audible artifacts.
    /// Range: [0.5, 20.0]. Default: 3.0.
    pub max_shift_ms: f32,
    /// Dry/wet mix: 0.0 = fully dry (bypass), 1.0 = fully aligned.
    /// Range: [0.0, 1.0]. Default: 0.5.
    pub mix: f32,
}

impl Default for TransientAlignConfig {
    fn default() -> Self {
        Self {
            detection_threshold_db: -30.0,
            alignment_strength: 0.5,
            lookahead_ms: 5.0,
            attack_window_ms: 10.0,
            max_shift_ms: 3.0,
            mix: 0.5,
        }
    }
}

impl TransientAlignConfig {
    /// Create a new config with all defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Tight alignment preset: snappy, mechanical precision.
    /// Good for pop/EDM backing vocals or rhythmic passages.
    #[must_use]
    pub fn tight() -> Self {
        Self {
            detection_threshold_db: -24.0,
            alignment_strength: 0.9,
            lookahead_ms: 5.0,
            attack_window_ms: 8.0,
            max_shift_ms: 5.0,
            mix: 0.8,
        }
    }

    /// Natural alignment preset: subtle tightening while preserving feel.
    /// Good for choir or ensemble passages that need a small nudge.
    #[must_use]
    pub fn natural() -> Self {
        Self {
            detection_threshold_db: -30.0,
            alignment_strength: 0.5,
            lookahead_ms: 5.0,
            attack_window_ms: 10.0,
            max_shift_ms: 3.0,
            mix: 0.5,
        }
    }

    /// Loose alignment preset: very gentle, barely perceptible.
    /// Good for ambient textures where timing variation is desirable.
    #[must_use]
    pub fn loose() -> Self {
        Self {
            detection_threshold_db: -36.0,
            alignment_strength: 0.25,
            lookahead_ms: 3.0,
            attack_window_ms: 15.0,
            max_shift_ms: 2.0,
            mix: 0.3,
        }
    }

    /// Percussion preset: ultra-tight for percussive consonants.
    /// Optimized for plosive-heavy text (t, k, p, d, g, b).
    #[must_use]
    pub fn percussion() -> Self {
        Self {
            detection_threshold_db: -20.0,
            alignment_strength: 0.95,
            lookahead_ms: 3.0,
            attack_window_ms: 5.0,
            max_shift_ms: 4.0,
            mix: 0.9,
        }
    }

    /// Set detection threshold in dB.
    #[must_use]
    pub fn with_detection_threshold_db(mut self, db: f32) -> Self {
        self.detection_threshold_db = db;
        self
    }

    /// Set alignment strength.
    #[must_use]
    pub fn with_alignment_strength(mut self, strength: f32) -> Self {
        self.alignment_strength = strength;
        self
    }

    /// Set look-ahead in milliseconds.
    #[must_use]
    pub fn with_lookahead_ms(mut self, ms: f32) -> Self {
        self.lookahead_ms = ms;
        self
    }

    /// Set attack window in milliseconds.
    #[must_use]
    pub fn with_attack_window_ms(mut self, ms: f32) -> Self {
        self.attack_window_ms = ms;
        self
    }

    /// Set maximum shift in milliseconds.
    #[must_use]
    pub fn with_max_shift_ms(mut self, ms: f32) -> Self {
        self.max_shift_ms = ms;
        self
    }

    /// Set dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix;
        self
    }

    /// Validate all configuration fields.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.detection_threshold_db.is_finite()
            || self.detection_threshold_db < -60.0
            || self.detection_threshold_db > -6.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "detection_threshold_db",
                reason: format!(
                    "must be finite and in [-60, -6], got {}",
                    self.detection_threshold_db,
                ),
            });
        }
        if !self.alignment_strength.is_finite() || !(0.0..=1.0).contains(&self.alignment_strength) {
            return Err(KokoroError::InvalidConfig {
                field: "alignment_strength",
                reason: format!(
                    "must be finite and in [0, 1], got {}",
                    self.alignment_strength,
                ),
            });
        }
        if !self.lookahead_ms.is_finite() || self.lookahead_ms < 0.5 || self.lookahead_ms > 20.0 {
            return Err(KokoroError::InvalidConfig {
                field: "lookahead_ms",
                reason: format!("must be finite and in [0.5, 20], got {}", self.lookahead_ms),
            });
        }
        if !self.attack_window_ms.is_finite()
            || self.attack_window_ms < 1.0
            || self.attack_window_ms > 50.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "attack_window_ms",
                reason: format!(
                    "must be finite and in [1, 50], got {}",
                    self.attack_window_ms,
                ),
            });
        }
        if !self.max_shift_ms.is_finite() || self.max_shift_ms < 0.5 || self.max_shift_ms > 20.0 {
            return Err(KokoroError::InvalidConfig {
                field: "max_shift_ms",
                reason: format!("must be finite and in [0.5, 20], got {}", self.max_shift_ms),
            });
        }
        if !self.mix.is_finite() || !(0.0..=1.0).contains(&self.mix) {
            return Err(KokoroError::InvalidConfig {
                field: "mix",
                reason: format!("must be finite and in [0, 1], got {}", self.mix),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Onset detection
// ---------------------------------------------------------------------------

/// A detected transient onset in a single voice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Onset {
    /// Sample index of the detected onset.
    pub sample: usize,
    /// Strength of the onset (energy derivative magnitude, >= 0).
    pub strength: f32,
}

/// Detect transient onsets in an audio buffer using energy derivative analysis.
///
/// Computes a short-time energy curve (window = `energy_window` samples),
/// takes its first difference, and marks positions where the derivative
/// exceeds `threshold_linear` as onsets.
///
/// Returns onset positions sorted by sample index.
fn detect_onsets(
    audio: &[f32],
    energy_window: usize,
    threshold_linear: f32,
    min_onset_gap: usize,
) -> Vec<Onset> {
    if audio.is_empty() || energy_window == 0 {
        return Vec::new();
    }
    let win = energy_window.min(audio.len());

    // Compute short-time energy using a sliding window.
    let n = audio.len();
    let n_frames = if n >= win { n - win + 1 } else { 1 };
    let mut energy = Vec::with_capacity(n_frames);

    // First window.
    let mut running_energy: f64 = audio[..win.min(n)]
        .iter()
        .map(|&s| {
            let s = if s.is_finite() { s } else { 0.0 };
            f64::from(s) * f64::from(s)
        })
        .sum();
    energy.push(running_energy as f32);

    // Slide the window.
    for i in 1..n_frames {
        let old = audio[i - 1];
        let old = if old.is_finite() { old } else { 0.0 };
        let new = audio[i + win - 1];
        let new = if new.is_finite() { new } else { 0.0 };
        running_energy += f64::from(new) * f64::from(new) - f64::from(old) * f64::from(old);
        if running_energy < 0.0 {
            running_energy = 0.0;
        }
        energy.push(running_energy as f32);
    }

    // Compute energy derivative (first difference).
    let mut onsets = Vec::new();
    let mut last_onset: Option<usize> = None;
    for i in 1..energy.len() {
        let deriv = energy[i] - energy[i - 1];
        if !deriv.is_finite() || deriv <= 0.0 {
            continue;
        }
        if deriv < threshold_linear {
            continue;
        }
        // Enforce minimum gap between onsets.
        if let Some(prev) = last_onset {
            if i - prev < min_onset_gap {
                continue;
            }
        }
        // Map frame index back to approximate sample index.
        let sample_idx = i + win / 2;
        if sample_idx < audio.len() {
            onsets.push(Onset {
                sample: sample_idx,
                strength: deriv,
            });
            last_onset = Some(i);
        }
    }
    onsets
}

// ---------------------------------------------------------------------------
// Circular look-ahead buffer
// ---------------------------------------------------------------------------

/// Small circular buffer used for look-ahead onset detection.
///
/// Stores the most recent `capacity` samples so the detector can look
/// slightly ahead of the current processing position.
#[derive(Debug, Clone)]
struct LookaheadBuffer {
    buf: Vec<f32>,
    write_pos: usize,
    capacity: usize,
}

impl LookaheadBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity.max(1)],
            write_pos: 0,
            capacity: capacity.max(1),
        }
    }

    /// Push a sample into the buffer, returning the sample that was evicted.
    #[inline]
    fn push(&mut self, sample: f32) -> f32 {
        let evicted = self.buf[self.write_pos];
        let s = if sample.is_finite() { sample } else { 0.0 };
        self.buf[self.write_pos] = s;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        evicted
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// TransientAligner
// ---------------------------------------------------------------------------

/// Stateful transient onset aligner for multi-voice chorus.
///
/// Detects transient attacks in each voice and micro-shifts them to a
/// common target time, producing tighter onset timing across the ensemble
/// while preserving natural sustain and release.
#[derive(Debug, Clone)]
pub struct TransientAligner {
    config: TransientAlignConfig,
    n_voices: usize,
    sample_rate: f32,
    /// Pre-computed sample counts.
    lookahead_samples: usize,
    attack_window_samples: usize,
    max_shift_samples: usize,
    energy_window_samples: usize,
    /// Detection threshold in linear energy-derivative units.
    threshold_linear: f32,
    /// Per-voice look-ahead buffers.
    lookahead_bufs: Vec<LookaheadBuffer>,
}

impl TransientAligner {
    /// Create a new transient aligner.
    ///
    /// # Errors
    ///
    /// Returns an error if the config is invalid, `n_voices` is zero, or
    /// `sample_rate` is non-positive / non-finite.
    pub fn new(
        config: &TransientAlignConfig,
        n_voices: usize,
        sample_rate: f32,
    ) -> Result<Self, KokoroError> {
        config.validate()?;
        if n_voices == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: "must be >= 1".to_string(),
            });
        }
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be finite and positive, got {sample_rate}"),
            });
        }

        let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);
        let attack_window_samples = ms_to_samples(config.attack_window_ms, sample_rate);
        let max_shift_samples = ms_to_samples(config.max_shift_ms, sample_rate);
        // Energy window: ~2ms for fine-grained detection.
        let energy_window_samples = ms_to_samples(2.0, sample_rate).max(2);

        // Convert dB threshold to linear energy-derivative scale.
        // threshold_db is negative (e.g. -30), so 10^(db/10) gives a small number.
        let threshold_linear = (10.0_f32).powf(config.detection_threshold_db / 10.0);
        let threshold_linear = if threshold_linear.is_finite() {
            threshold_linear
        } else {
            1e-3
        };

        let lookahead_bufs = (0..n_voices)
            .map(|_| LookaheadBuffer::new(lookahead_samples))
            .collect();

        Ok(Self {
            config: *config,
            n_voices,
            sample_rate,
            lookahead_samples,
            attack_window_samples,
            max_shift_samples,
            energy_window_samples,
            threshold_linear,
            lookahead_bufs,
        })
    }

    /// Process multiple voices, aligning their transient onsets in-place.
    ///
    /// All voice buffers should be the same length. Voices shorter than the
    /// longest are zero-padded internally; voices longer are truncated to
    /// the minimum length.
    ///
    /// With fewer than 2 voices or mix == 0, this is a no-op.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) -> Result<(), KokoroError> {
        if voices.len() < 2 || self.config.mix <= 0.0 {
            return Ok(());
        }
        if voices.len() != self.n_voices {
            return Err(KokoroError::InvalidInput(format!(
                "expected {} voices, got {}",
                self.n_voices,
                voices.len(),
            )));
        }

        // Sanitize NaN/Inf in all voice buffers (defense-in-depth).
        for voice in voices.iter_mut() {
            for sample in voice.iter_mut() {
                if !sample.is_finite() {
                    *sample = 0.0;
                }
            }
        }

        // Find the common length (minimum across all voices).
        let common_len = voices.iter().map(Vec::len).min().unwrap_or(0);
        if common_len == 0 {
            return Ok(());
        }

        // Minimum onset gap: half of attack window to prevent double-triggers.
        let min_onset_gap = self.attack_window_samples / 2;

        // Step 1: Detect onsets in each voice.
        let per_voice_onsets: Vec<Vec<Onset>> = voices
            .iter()
            .map(|v| {
                detect_onsets(
                    &v[..common_len],
                    self.energy_window_samples,
                    self.threshold_linear,
                    min_onset_gap.max(1),
                )
            })
            .collect();

        // Step 2: Match onsets across voices and compute alignment shifts.
        // For each onset cluster (onsets within max_shift_samples of each
        // other across voices), compute the earliest onset as the target.
        let clusters = cluster_onsets(&per_voice_onsets, self.max_shift_samples);

        // Step 3: Apply micro-shifts to each voice's attack windows.
        let mix = self.config.mix;
        let strength = self.config.alignment_strength;

        for cluster in &clusters {
            // Find the earliest onset time in the cluster.
            let earliest = cluster
                .iter()
                .map(|&(_, onset)| onset.sample)
                .min()
                .unwrap_or(0);

            for &(voice_idx, onset) in cluster {
                if voice_idx >= voices.len() {
                    continue;
                }
                let voice = &mut voices[voice_idx];
                let shift_samples = onset.sample as i64 - earliest as i64;

                if shift_samples == 0 {
                    continue;
                }

                // Effective shift with alignment strength and mix applied.
                let effective_shift = (shift_samples as f32 * strength * mix).round() as i64;
                if effective_shift == 0 {
                    continue;
                }

                // Apply micro-shift to the attack window around the onset.
                apply_attack_shift(
                    voice,
                    onset.sample,
                    effective_shift,
                    self.attack_window_samples,
                    common_len,
                );
            }
        }

        Ok(())
    }

    /// Reset all internal state (look-ahead buffers).
    pub fn reset(&mut self) {
        for buf in &mut self.lookahead_bufs {
            buf.reset();
        }
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &TransientAlignConfig {
        &self.config
    }

    /// Number of voices this aligner was configured for.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.n_voices
    }

    /// Sample rate this aligner was configured for.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Onset clustering
// ---------------------------------------------------------------------------

/// Cluster onsets across voices that fall within `max_gap` samples of
/// each other. Returns groups of (voice_index, onset) tuples.
fn cluster_onsets(per_voice: &[Vec<Onset>], max_gap: usize) -> Vec<Vec<(usize, Onset)>> {
    // Flatten all onsets with voice index, sorted by sample position.
    let mut all_onsets: Vec<(usize, Onset)> = Vec::new();
    for (vi, onsets) in per_voice.iter().enumerate() {
        for &onset in onsets {
            all_onsets.push((vi, onset));
        }
    }
    all_onsets.sort_by_key(|&(_, o)| o.sample);

    // Greedy clustering: consecutive onsets within max_gap go into one cluster.
    let mut clusters: Vec<Vec<(usize, Onset)>> = Vec::new();
    let mut current_cluster: Vec<(usize, Onset)> = Vec::new();

    for entry in &all_onsets {
        if let Some(last) = current_cluster.last() {
            if entry.1.sample.saturating_sub(last.1.sample) > max_gap {
                if current_cluster.len() >= 2 {
                    clusters.push(std::mem::take(&mut current_cluster));
                } else {
                    current_cluster.clear();
                }
            }
        }
        current_cluster.push(*entry);
    }
    // Don't forget the last cluster.
    if current_cluster.len() >= 2 {
        clusters.push(current_cluster);
    }

    clusters
}

// ---------------------------------------------------------------------------
// Attack micro-shift
// ---------------------------------------------------------------------------

/// Shift the attack window around `onset_sample` by `shift` samples
/// (negative = move earlier). Uses crossfade at window boundaries to
/// avoid clicks.
fn apply_attack_shift(
    voice: &mut [f32],
    onset_sample: usize,
    shift: i64,
    attack_window: usize,
    max_len: usize,
) {
    let half_win = attack_window / 2;
    let win_start = onset_sample.saturating_sub(half_win);
    let win_end = (onset_sample + half_win).min(max_len).min(voice.len());
    if win_start >= win_end {
        return;
    }

    let win_len = win_end - win_start;
    // Crossfade length: 10% of window, minimum 1 sample.
    let fade_len = (win_len / 10).max(1);

    // Extract the window.
    let original_window: Vec<f32> = voice[win_start..win_end].to_vec();
    let mut shifted_window = vec![0.0f32; win_len];

    for i in 0..win_len {
        let src = i as i64 - shift;
        if src >= 0 && (src as usize) < win_len {
            let val = original_window[src as usize];
            shifted_window[i] = if val.is_finite() { val } else { 0.0 };
        }
        // Out-of-range samples stay at 0 — crossfade will blend them.
    }

    // Apply raised-cosine crossfade at window boundaries.
    for i in 0..win_len {
        // Crossfade envelope: ramp in at start, ramp out at end.
        let envelope = if i < fade_len {
            let t = i as f32 / fade_len as f32;
            0.5 * (1.0 - (std::f32::consts::PI * t).cos())
        } else if i >= win_len - fade_len {
            let t = (win_len - 1 - i) as f32 / fade_len as f32;
            0.5 * (1.0 - (std::f32::consts::PI * t).cos())
        } else {
            1.0
        };
        let envelope = if envelope.is_finite() { envelope } else { 0.0 };

        let orig = original_window[i];
        let orig = if orig.is_finite() { orig } else { 0.0 };
        let shifted = shifted_window[i];
        let shifted = if shifted.is_finite() { shifted } else { 0.0 };

        // Blend: envelope controls how much of the shifted signal replaces
        // the original within the window.
        let blended = orig * (1.0 - envelope) + shifted * envelope;
        voice[win_start + i] = if blended.is_finite() { blended } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert milliseconds to samples, clamped to at least 1.
fn ms_to_samples(ms: f32, sample_rate: f32) -> usize {
    let s = (ms * 0.001 * sample_rate).round() as usize;
    s.max(1)
}

#[cfg(test)]
#[path = "kokoro_chorus_transient_align_tests.rs"]
mod tests;
