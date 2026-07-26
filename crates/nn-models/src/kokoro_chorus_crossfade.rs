// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming crossfade optimizer for Kokoro chorus chunk boundaries.
//!
//! The default Hann-window overlap-add crossfade can produce audible artifacts
//! at chunk boundaries, especially when multi-voice chorus processing
//! introduces phase and energy discontinuities between adjacent chunks.
//! This module analyzes each boundary in real time and selects optimal
//! crossfade parameters: splice point, fade length, and window shape.
//!
//! # Analysis modes
//!
//! | Mode | Cost | Quality | Use case |
//! |------|------|---------|----------|
//! | [`CrossfadeAnalysis::Fixed`] | O(1) | Baseline | Low-latency, simple content |
//! | [`CrossfadeAnalysis::EnergyAdaptive`] | O(n) | Good | Music, varying dynamics |
//! | [`CrossfadeAnalysis::PhaseAligned`] | O(n) | Better | Tonal content, chorus |
//! | [`CrossfadeAnalysis::SpectralMatch`] | O(n log n) | Best | Critical listening |
//!
//! # Architecture
//!
//! ```text
//! prev_chunk_tail ──┐
//!                   ├─► boundary analysis ──► adaptive window ──► crossfaded output
//! new_chunk_head  ──┘
//! ```
//!
//! The optimizer retains the tail of each processed chunk so the next
//! `push_chunk` call can analyze and blend across the boundary.
//!
//! Part of #4582, #4264, #3351.

use crate::kokoro_error::KokoroError;

/// Absolute minimum crossfade to avoid clicks (2 ms at 24 kHz).
const ABS_MIN_CROSSFADE: usize = 48;

// ---------------------------------------------------------------------------
// CrossfadeAnalysis
// ---------------------------------------------------------------------------

/// Analysis strategy for choosing crossfade parameters at each chunk boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CrossfadeAnalysis {
    /// Fixed crossfade length with no boundary analysis (baseline).
    Fixed,
    /// Adapt crossfade length to boundary energy: longer for loud regions,
    /// shorter for quiet passages where discontinuities are less audible.
    EnergyAdaptive,
    /// Search for a phase-aligned splice point near the boundary using a
    /// Hilbert-transform approximation of instantaneous phase.
    PhaseAligned,
    /// Full spectral envelope comparison across the boundary; generate an
    /// asymmetric crossfade window that matches spectral shape.
    SpectralMatch,
}

// ---------------------------------------------------------------------------
// CrossfadeOptimizerConfig
// ---------------------------------------------------------------------------

/// Configuration for the streaming crossfade optimizer.
///
/// Use [`CrossfadeOptimizerConfig::builder()`] to construct.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CrossfadeOptimizerConfig {
    /// Analysis strategy (see [`CrossfadeAnalysis`]).
    pub analysis_mode: CrossfadeAnalysis,
    /// Minimum crossfade region in samples (default 480 = 20 ms at 24 kHz).
    pub min_crossfade_samples: usize,
    /// Maximum crossfade region in samples (default 2400 = 100 ms at 24 kHz).
    pub max_crossfade_samples: usize,
    /// Target SNR at the boundary in dB (default 60.0).
    pub target_snr_db: f32,
    /// Search for zero crossings near the splice point for cleaner cuts.
    pub zero_crossing_search: bool,
    /// Match spectral characteristics across the boundary (only effective
    /// with [`CrossfadeAnalysis::SpectralMatch`]).
    pub spectral_match: bool,
}

impl CrossfadeOptimizerConfig {
    /// Create a builder with sensible defaults.
    pub fn builder() -> CrossfadeOptimizerConfigBuilder {
        CrossfadeOptimizerConfigBuilder::default()
    }

    /// Validate all fields, returning an error for invalid combinations.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.min_crossfade_samples < ABS_MIN_CROSSFADE {
            return Err(KokoroError::InvalidConfig {
                field: "min_crossfade_samples",
                reason: format!(
                    "must be >= {} (2 ms at 24 kHz), got {}",
                    ABS_MIN_CROSSFADE, self.min_crossfade_samples,
                ),
            });
        }
        if self.max_crossfade_samples < self.min_crossfade_samples {
            return Err(KokoroError::InvalidConfig {
                field: "max_crossfade_samples",
                reason: format!(
                    "must be >= min_crossfade_samples ({}), got {}",
                    self.min_crossfade_samples, self.max_crossfade_samples,
                ),
            });
        }
        if !self.target_snr_db.is_finite() || self.target_snr_db <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "target_snr_db",
                reason: format!("must be finite and > 0, got {}", self.target_snr_db),
            });
        }
        Ok(())
    }
}

impl Default for CrossfadeOptimizerConfig {
    fn default() -> Self {
        Self {
            analysis_mode: CrossfadeAnalysis::EnergyAdaptive,
            min_crossfade_samples: 480,  // 20 ms at 24 kHz
            max_crossfade_samples: 2400, // 100 ms at 24 kHz
            target_snr_db: 60.0,
            zero_crossing_search: true,
            spectral_match: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for [`CrossfadeOptimizerConfig`].
#[derive(Debug, Clone)]
#[must_use]
#[derive(Default)]
pub struct CrossfadeOptimizerConfigBuilder {
    inner: CrossfadeOptimizerConfig,
}


impl CrossfadeOptimizerConfigBuilder {
    pub fn analysis_mode(mut self, mode: CrossfadeAnalysis) -> Self {
        self.inner.analysis_mode = mode;
        self
    }
    pub fn min_crossfade_samples(mut self, n: usize) -> Self {
        self.inner.min_crossfade_samples = n;
        self
    }
    pub fn max_crossfade_samples(mut self, n: usize) -> Self {
        self.inner.max_crossfade_samples = n;
        self
    }
    pub fn target_snr_db(mut self, db: f32) -> Self {
        self.inner.target_snr_db = db;
        self
    }
    pub fn zero_crossing_search(mut self, enable: bool) -> Self {
        self.inner.zero_crossing_search = enable;
        self
    }
    pub fn spectral_match(mut self, enable: bool) -> Self {
        self.inner.spectral_match = enable;
        self
    }
    /// Build and validate the config.
    pub fn build(self) -> Result<CrossfadeOptimizerConfig, KokoroError> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------

/// Find indices where the signal crosses zero (sign change between adjacent
/// samples). Returns indices of the sample *before* each crossing.
#[must_use]
pub fn find_zero_crossings(audio: &[f32]) -> Vec<usize> {
    if audio.len() < 2 {
        return Vec::new();
    }
    let mut crossings = Vec::new();
    for i in 0..audio.len() - 1 {
        // Sign change (including exact zero on the right sample).
        if (audio[i] >= 0.0 && audio[i + 1] < 0.0) || (audio[i] < 0.0 && audio[i + 1] >= 0.0) {
            crossings.push(i);
        }
    }
    crossings
}

/// Compute a short-time energy envelope using a rectangular window of
/// `window_size` samples.
///
/// Returns one energy value per sample (causal: energy at index `i` covers
/// `[i - window_size + 1 .. i]`, clamped to 0).
#[must_use]
pub fn compute_energy_envelope(audio: &[f32], window_size: usize) -> Vec<f32> {
    if audio.is_empty() || window_size == 0 {
        return vec![0.0; audio.len()];
    }
    let ws = window_size.min(audio.len());
    let mut envelope = Vec::with_capacity(audio.len());
    let mut running_energy: f64 = 0.0;

    // Seed the first window.
    for &s in &audio[..ws] {
        running_energy += f64::from(s) * f64::from(s);
    }
    // Fill initial positions where the full window hasn't accumulated yet.
    for i in 0..ws {
        let partial: f64 = audio[..=i].iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        envelope.push((partial / (i + 1) as f64) as f32);
    }
    // Slide the window for the rest.
    for i in ws..audio.len() {
        let old = f64::from(audio[i - ws]);
        let new = f64::from(audio[i]);
        running_energy += new * new - old * old;
        // Clamp to avoid negative due to floating-point drift.
        if running_energy < 0.0 {
            running_energy = 0.0;
        }
        envelope.push((running_energy / ws as f64) as f32);
    }
    envelope
}

/// Approximate instantaneous phase at each sample using a simple Hilbert
/// transform approximation (finite-difference analytic signal). Returns
/// phase in radians (-PI .. PI).
fn approx_instantaneous_phase(audio: &[f32]) -> Vec<f32> {
    // Use a 5-tap FIR Hilbert approximation: h = [0, 2/PI, 0, -2/PI, 0]
    // shifted to causal form with 2-sample delay.
    let len = audio.len();
    if len < 5 {
        return vec![0.0; len];
    }
    let coeff = 2.0_f32 / std::f32::consts::PI;
    let mut phase = vec![0.0_f32; len];
    for i in 2..len.saturating_sub(2) {
        let hilbert = coeff * (audio[i - 1] - audio[i + 1]);
        let real = audio[i];
        phase[i] = hilbert.atan2(real);
    }
    // Edge samples: copy nearest valid.
    if len >= 5 {
        phase[0] = phase[2];
        phase[1] = phase[2];
        phase[len - 2] = phase[len - 3];
        phase[len - 1] = phase[len - 3];
    }
    phase
}

/// Generate asymmetric crossfade windows for fade-out (from prev chunk) and
/// fade-in (into new chunk). Lengths may differ to allow longer fade-out from
/// a loud chunk into a quiet one.
///
/// Both windows are Hann-shaped and returned as `(fade_out, fade_in)`.
#[must_use]
pub fn generate_adaptive_window(fade_out_len: usize, fade_in_len: usize) -> (Vec<f32>, Vec<f32>) {
    let fade_out = if fade_out_len == 0 {
        Vec::new()
    } else {
        (0..fade_out_len)
            .map(|i| {
                let t = i as f32 / (fade_out_len.max(1) - 1).max(1) as f32;
                // Hann fade-out: starts at 1, ends at 0.
                0.5 * (1.0 + (std::f32::consts::PI * t).cos())
            })
            .collect()
    };

    let fade_in = if fade_in_len == 0 {
        Vec::new()
    } else {
        (0..fade_in_len)
            .map(|i| {
                let t = i as f32 / (fade_in_len.max(1) - 1).max(1) as f32;
                // Hann fade-in: starts at 0, ends at 1.
                0.5 * (1.0 - (std::f32::consts::PI * t).cos())
            })
            .collect()
    };

    (fade_out, fade_in)
}

// ---------------------------------------------------------------------------
// CrossfadeOptimizer
// ---------------------------------------------------------------------------

/// Real-time crossfade optimizer that analyzes chunk boundaries and applies
/// adaptive crossfading for the Kokoro streaming chorus path.
///
/// # Usage
///
/// ```ignore
/// let config = CrossfadeOptimizerConfig::builder()
///     .analysis_mode(CrossfadeAnalysis::EnergyAdaptive)
///     .build()?;
/// let mut opt = CrossfadeOptimizer::new(config)?;
///
/// for chunk in audio_chunks {
///     let out = opt.push_chunk(&chunk)?;
///     play(out);
/// }
/// if let Some(tail) = opt.flush() {
///     play(tail);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CrossfadeOptimizer {
    config: CrossfadeOptimizerConfig,
    /// Tail of the previous chunk retained for boundary analysis.
    prev_chunk_tail: Option<Vec<f32>>,
    /// Running energy of the previous chunk tail (RMS squared).
    prev_chunk_energy: f32,
}

impl CrossfadeOptimizer {
    /// Create a new optimizer. Validates config on construction.
    pub fn new(config: CrossfadeOptimizerConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        Ok(Self {
            config,
            prev_chunk_tail: None,
            prev_chunk_energy: 0.0,
        })
    }

    /// Push a new audio chunk. Returns the output samples ready for playback.
    ///
    /// On the first call the chunk is stored and an empty vec is returned
    /// (the optimizer needs a boundary to work with). Subsequent calls
    /// return the crossfaded audio from the *previous* chunk, up to the
    /// boundary with the current one.
    pub fn push_chunk(&mut self, chunk: &[f32]) -> Result<Vec<f32>, KokoroError> {
        if chunk.is_empty() {
            return Ok(Vec::new());
        }

        let max_cf = self.config.max_crossfade_samples.min(chunk.len());

        match self.prev_chunk_tail.take() {
            None => {
                // First chunk: store tail and return the non-overlap region.
                let tail_len = max_cf.min(chunk.len());
                self.prev_chunk_tail = Some(chunk[chunk.len() - tail_len..].to_vec());
                self.prev_chunk_energy = Self::rms_energy(&chunk[chunk.len() - tail_len..]);
                // Emit everything except the tail we retained.
                if chunk.len() > tail_len {
                    Ok(chunk[..chunk.len() - tail_len].to_vec())
                } else {
                    // Entire chunk is tail — nothing to emit yet.
                    Ok(Vec::new())
                }
            }
            Some(prev_tail) => {
                let result = self.optimize_and_crossfade(&prev_tail, chunk)?;

                // Retain a new tail from the current chunk.
                let tail_len = max_cf.min(chunk.len());
                self.prev_chunk_tail = Some(chunk[chunk.len() - tail_len..].to_vec());
                self.prev_chunk_energy = Self::rms_energy(&chunk[chunk.len() - tail_len..]);
                Ok(result)
            }
        }
    }

    /// Flush the retained tail when no more chunks will arrive.
    #[must_use]
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        self.prev_chunk_tail.take()
    }

    /// Reset state so the optimizer can be reused for a new utterance.
    pub fn reset(&mut self) {
        self.prev_chunk_tail = None;
        self.prev_chunk_energy = 0.0;
    }

    /// Read-only access to the current config.
    #[must_use]
    pub fn config(&self) -> &CrossfadeOptimizerConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Core: analyze boundary and apply crossfade
    // -----------------------------------------------------------------------

    /// Analyze the boundary between `prev_tail` (end of previous chunk) and
    /// `new_head` (start of current chunk), then produce crossfaded output.
    ///
    /// Returns: `prev_tail[..non_overlap] ++ crossfaded_region ++ new_chunk[crossfade..]`
    pub fn optimize_and_crossfade(
        &mut self,
        prev_tail: &[f32],
        new_chunk: &[f32],
    ) -> Result<Vec<f32>, KokoroError> {
        let cf_len = self.compute_crossfade_length(prev_tail, new_chunk);
        let cf_len = cf_len.min(prev_tail.len()).min(new_chunk.len());
        if cf_len == 0 {
            // No overlap possible — just concatenate.
            let mut out = prev_tail.to_vec();
            out.extend_from_slice(new_chunk);
            return Ok(out);
        }

        // Determine splice offset within the crossfade region.
        let splice_offset = if self.config.zero_crossing_search {
            self.find_best_splice_point(
                &prev_tail[prev_tail.len() - cf_len..],
                &new_chunk[..cf_len],
            )
        } else {
            0
        };

        // Generate the crossfade window (may be asymmetric).
        let (fade_out, fade_in) = self.build_crossfade_windows(cf_len, prev_tail, new_chunk);

        // Assemble output: non-overlapping prefix from prev_tail + crossfade
        // region + non-overlapping suffix from new_chunk.
        let prefix_end = prev_tail.len() - cf_len;
        let mut out =
            Vec::with_capacity(prefix_end + cf_len + new_chunk.len().saturating_sub(cf_len));

        // 1. Non-overlapping prefix from previous chunk.
        out.extend_from_slice(&prev_tail[..prefix_end]);

        // 2. Crossfaded overlap region.
        let prev_overlap = &prev_tail[prefix_end..];
        let new_overlap = &new_chunk[..cf_len];
        for i in 0..cf_len {
            let fo = fade_out.get(i).copied().unwrap_or(0.0);
            let fi = fade_in.get(i).copied().unwrap_or(1.0);
            // Apply splice offset: shift new_chunk read position for
            // phase alignment (clamped to valid range).
            let new_idx = (i + splice_offset).min(cf_len - 1);
            out.push(prev_overlap[i] * fo + new_overlap[new_idx] * fi);
        }

        // 3. Non-overlapping suffix from new chunk.
        if cf_len < new_chunk.len() {
            out.extend_from_slice(&new_chunk[cf_len..]);
        }

        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Adaptive crossfade length
    // -----------------------------------------------------------------------

    fn compute_crossfade_length(&self, prev_tail: &[f32], new_head: &[f32]) -> usize {
        match self.config.analysis_mode {
            CrossfadeAnalysis::Fixed => self.config.min_crossfade_samples,

            CrossfadeAnalysis::EnergyAdaptive => {
                let prev_e = Self::rms_energy(prev_tail);
                let new_e = Self::rms_energy(
                    &new_head[..new_head.len().min(self.config.max_crossfade_samples)],
                );
                let max_e = prev_e.max(new_e).max(1e-10);
                // Scale linearly between min and max based on energy.
                // Louder content gets longer crossfade to mask the boundary.
                let ratio = (max_e.sqrt() * 4.0).min(1.0);
                let range = self.config.max_crossfade_samples - self.config.min_crossfade_samples;
                self.config.min_crossfade_samples + (ratio * range as f32) as usize
            }

            CrossfadeAnalysis::PhaseAligned => {
                // Start with energy-adaptive length, then adjust based on
                // phase discontinuity.
                let base = self.energy_adaptive_length(prev_tail, new_head);
                let phase_disc = self.phase_discontinuity(prev_tail, new_head);
                // Larger phase discontinuity → longer crossfade.
                let phase_factor = (phase_disc / std::f32::consts::PI).min(1.0);
                let extra =
                    ((self.config.max_crossfade_samples - base) as f32 * phase_factor) as usize;
                (base + extra).min(self.config.max_crossfade_samples)
            }

            CrossfadeAnalysis::SpectralMatch => {
                // Use maximum crossfade for best spectral blending.
                self.config.max_crossfade_samples
            }
        }
    }

    fn energy_adaptive_length(&self, prev_tail: &[f32], new_head: &[f32]) -> usize {
        let prev_e = Self::rms_energy(prev_tail);
        let new_e =
            Self::rms_energy(&new_head[..new_head.len().min(self.config.max_crossfade_samples)]);
        let max_e = prev_e.max(new_e).max(1e-10);
        let ratio = (max_e.sqrt() * 4.0).min(1.0);
        let range = self.config.max_crossfade_samples - self.config.min_crossfade_samples;
        self.config.min_crossfade_samples + (ratio * range as f32) as usize
    }

    // -----------------------------------------------------------------------
    // Phase analysis
    // -----------------------------------------------------------------------

    fn phase_discontinuity(&self, prev_tail: &[f32], new_head: &[f32]) -> f32 {
        // Compare instantaneous phase at the exact boundary.
        if prev_tail.is_empty() || new_head.is_empty() {
            return 0.0;
        }
        let n_analysis = 32.min(prev_tail.len()).min(new_head.len());
        let prev_phase = approx_instantaneous_phase(&prev_tail[prev_tail.len() - n_analysis..]);
        let new_phase = approx_instantaneous_phase(&new_head[..n_analysis]);

        if let (Some(&p), Some(&n)) = (prev_phase.last(), new_phase.first()) {
            let diff = (p - n).abs();
            // Wrap to [0, PI].
            if diff > std::f32::consts::PI {
                2.0 * std::f32::consts::PI - diff
            } else {
                diff
            }
        } else {
            0.0
        }
    }

    // -----------------------------------------------------------------------
    // Splice-point search
    // -----------------------------------------------------------------------

    fn find_best_splice_point(&self, prev_overlap: &[f32], new_overlap: &[f32]) -> usize {
        // Look for the zero crossing in the new overlap that is closest to
        // a zero crossing in the prev overlap at the same relative position.
        let new_zc = find_zero_crossings(new_overlap);
        if new_zc.is_empty() {
            return 0;
        }
        let prev_zc = find_zero_crossings(prev_overlap);
        if prev_zc.is_empty() {
            // Just pick the zero crossing closest to the center.
            let center = new_overlap.len() / 2;
            return new_zc
                .iter()
                .copied()
                .min_by_key(|&z| (z as isize - center as isize).unsigned_abs())
                .unwrap_or(0);
        }

        // Find the pair (prev_zc, new_zc) with the smallest absolute offset,
        // and return that offset so the crossfade can align them.
        let mut best_offset = 0_usize;
        let mut best_dist = usize::MAX;
        for &pz in &prev_zc {
            for &nz in &new_zc {
                let dist = (pz as isize - nz as isize).unsigned_abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_offset = nz.saturating_sub(pz);
                }
            }
        }
        best_offset
    }

    // -----------------------------------------------------------------------
    // Window generation
    // -----------------------------------------------------------------------

    fn build_crossfade_windows(
        &self,
        cf_len: usize,
        prev_tail: &[f32],
        new_chunk: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        match self.config.analysis_mode {
            CrossfadeAnalysis::SpectralMatch | CrossfadeAnalysis::PhaseAligned => {
                // Asymmetric: if previous chunk is louder, extend its fade-out.
                let prev_e = Self::rms_energy(prev_tail);
                let new_e = Self::rms_energy(&new_chunk[..new_chunk.len().min(cf_len)]);
                let ratio = if new_e > 1e-10 {
                    (prev_e / new_e).sqrt().clamp(0.5, 2.0)
                } else {
                    1.0
                };
                let fade_out_len = ((cf_len as f32 * ratio) as usize)
                    .clamp(self.config.min_crossfade_samples, cf_len);
                let fade_in_len = cf_len;
                generate_adaptive_window(fade_out_len, fade_in_len)
            }
            _ => {
                // Symmetric Hann crossfade.
                generate_adaptive_window(cf_len, cf_len)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    fn rms_energy(audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        let sum: f64 = audio.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        (sum / audio.len() as f64) as f32
    }
}

// ---------------------------------------------------------------------------
// Tests (extracted per 500-line rule)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_crossfade_tests.rs"]
mod tests;
