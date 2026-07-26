// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic chorus types and audio mixing for multi-voice Kokoro TTS.
//!
//! This module provides the data types and mixing logic shared between the
//! CPU (`KokoroModel`) and GPU (`CompiledKokoro` / `KokoroChorus`) paths.
//!
//! # Architecture
//!
//! ```text
//! N voices × (phonemes, style, speed)
//!   → N × synthesize()     // each voice independently
//!   → N × Vec<f32>         // raw PCM per voice
//!   → mix_voices_with_config()  // gain-weighted sum, optional stereo + clipping
//!   → Vec<f32>             // final mixed PCM
//! ```
//!
//! # Integration with streaming
//!
//! Each voice can use chunked streaming ([`assemble_streaming_chunks`]) for
//! first-audio latency. Chorus mixing happens after chunk assembly — either
//! per-chunk (low-latency) or on the full concatenated audio (simpler).
//!
//! Part of #3355, #3351, #2740.

use crate::kokoro_chorus_reverb::{apply_reverb, ReverbConfig};
use crate::kokoro_chorus_stereo::{default_voice_layout, StereoChorusConfig, StereoPosition};
use crate::kokoro_error::{validate_speed, KokoroError};
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Chorus configuration
// ---------------------------------------------------------------------------

/// Configuration for a multi-voice chorus synthesis pool.
///
/// Controls how many voices are synthesized and how their audio is mixed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ChorusConfig {
    /// Number of voices in the chorus (1–32).
    ///
    /// Each voice gets its own `CompiledKokoro` instance (via `clone_dispatch()`)
    /// sharing model weights through `Arc`. Memory overhead: ~1.02x per voice
    /// (segment caches only, weights are aliased).
    pub n_voices: usize,

    /// Per-voice gain multiplier applied before mixing.
    ///
    /// Length must equal `n_voices`. Each value must be in [0.0, 1.0].
    /// Typically 1.0/n_voices for equal-energy mixing, but can be adjusted
    /// for voice prominence (e.g., lead voice at 0.6, others at 0.2).
    pub gains: Vec<f32>,

    /// Whether to clip the mixed output to [-1.0, 1.0].
    ///
    /// Default: `true`. Prevents DAC overflow from constructive interference
    /// when multiple voices are in phase.
    pub clip_output: bool,

    /// Optional per-voice stereo pan positions in [-1.0, 1.0].
    ///
    /// When `Some`, chorus methods use [`mix_voices_stereo`] instead of
    /// [`mix_voices`]. Length must equal `n_voices`.
    /// -1.0 = hard left, 0.0 = center, 1.0 = hard right.
    pub pans: Option<Vec<f32>>,

    /// Per-voice pitch shift in semitones.
    ///
    /// Length must equal `n_voices` when `Some`. Each value is in [-12.0, 12.0]
    /// semitones. Even small detuning (±0.05 to ±0.10 semitones = ±5-10 cents)
    /// creates a richer, more natural chorus effect by preventing robotic unison.
    ///
    /// Applied as an F0 contour scale factor: `f0 * 2^(semitones/12)`.
    /// Default: `None` (no pitch shift).
    pub pitch_semitones: Option<Vec<f32>>,

    /// Per-voice timing offset in seconds.
    ///
    /// Length must equal `n_voices` when `Some`. Each value is in
    /// [-0.050, 0.050] (±50ms). Small offsets (±2-5ms) break up the
    /// "robotic unison" effect where all voices attack simultaneously.
    ///
    /// Positive values delay the voice; negative values advance it.
    /// Applied as sample-level shift in the mixed output.
    /// Default: `None` (no timing offset).
    pub timing_offsets_sec: Option<Vec<f32>>,

    /// Stereo width control for pan spread.
    ///
    /// Range: [0.0, 1.0]. Scales the effective pan positions:
    /// `effective_pan = pan * stereo_width`. At 0.0 all voices collapse
    /// to center (mono). At 1.0 voices use their full pan positions.
    ///
    /// Default: `1.0` (full stereo width). Only affects stereo output
    /// (when `pans` is `Some`).
    pub stereo_width: f32,

    /// Soft limiter drive gain for tanh-style saturation.
    ///
    /// When `Some(drive)`, applies `tanh(x * drive) / drive` instead of
    /// hard clipping. This rounds off peaks smoothly, avoiding the harsh
    /// distortion of hard clipping at ±1.0.
    ///
    /// Typical values: 1.0-2.0. At `drive = 1.0` the limiter is gentle
    /// (tanh is nearly linear for small signals). At `drive = 2.0` the
    /// limiter engages earlier, providing more compression.
    ///
    /// Default: `None` (use hard clip when `clip_output` is true).
    /// When set, takes precedence over `clip_output`.
    pub soft_limiter_drive: Option<f32>,

    /// Use `1/sqrt(n_voices)` gain normalization instead of `1/n_voices`.
    ///
    /// For uncorrelated voices, `1/sqrt(N)` preserves perceived loudness
    /// better than `1/N` which sounds quieter as voice count increases.
    /// This is the standard equal-power normalization for additive mixing.
    ///
    /// Default: `false` (use `1/n_voices` for backward compatibility).
    /// When `true`, `equal_gain()` and related constructors use `1/sqrt(N)`.
    pub sqrt_gain_normalization: bool,

    /// Optional spatial reverb configuration.
    ///
    /// When `Some`, applies a Schroeder reverb (4 comb + 2 allpass filters)
    /// with optional per-voice early reflections after mixing. Creates
    /// choir-like spatial depth — voices sound like they are in a real room.
    ///
    /// Default: `None` (no reverb). Use [`with_reverb`](Self::with_reverb)
    /// to enable with default settings, or pass a custom [`ReverbConfig`].
    pub reverb: Option<ReverbConfig>,
}

impl ChorusConfig {
    /// Create a chorus config with equal gains for all voices.
    ///
    /// Each voice gets gain = `1.0 / n_voices`, so the mixed output
    /// stays in [-1.0, 1.0] without clipping for uncorrelated signals.
    pub fn equal_gain(n_voices: usize) -> Result<Self, KokoroError> {
        if n_voices == 0 || n_voices > 32 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: format!("must be 1..=32, got {n_voices}"),
            });
        }
        let gain = 1.0 / n_voices as f32;
        Ok(Self {
            n_voices,
            gains: vec![gain; n_voices],
            clip_output: true,
            pans: None,
            pitch_semitones: None,
            timing_offsets_sec: None,
            stereo_width: 1.0,
            soft_limiter_drive: None,
            sqrt_gain_normalization: false,
            reverb: None,
        })
    }

    /// Create a chorus config with custom per-voice gains.
    pub fn with_gains(gains: Vec<f32>) -> Result<Self, KokoroError> {
        let n_voices = gains.len();
        if n_voices == 0 || n_voices > 32 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: format!("gains length must be 1..=32, got {n_voices}"),
            });
        }
        for (i, &g) in gains.iter().enumerate() {
            if !g.is_finite() || !(0.0..=1.0).contains(&g) {
                return Err(KokoroError::InvalidConfig {
                    field: "gains",
                    reason: format!("gain[{i}] = {g}: must be finite and in [0.0, 1.0]"),
                });
            }
        }
        Ok(Self {
            n_voices,
            gains,
            clip_output: true,
            pans: None,
            pitch_semitones: None,
            timing_offsets_sec: None,
            stereo_width: 1.0,
            soft_limiter_drive: None,
            sqrt_gain_normalization: false,
            reverb: None,
        })
    }

    /// Create a chorus config with per-voice stereo pan positions.
    ///
    /// Pans are in [-1.0, 1.0]: -1.0 = hard left, 0.0 = center, 1.0 = hard right.
    /// When pans are set, [`mix_voices_with_config`] uses [`mix_voices_stereo`]
    /// instead of [`mix_voices`], producing interleaved stereo output.
    pub fn with_stereo_pan(gains: Vec<f32>, pans: Vec<f32>) -> Result<Self, KokoroError> {
        if gains.len() != pans.len() {
            return Err(KokoroError::InvalidInput(format!(
                "gains length {} != pans length {}",
                gains.len(),
                pans.len(),
            )));
        }
        for (i, &p) in pans.iter().enumerate() {
            if !p.is_finite() || !(-1.0..=1.0).contains(&p) {
                return Err(KokoroError::InvalidConfig {
                    field: "pans",
                    reason: format!("pan[{i}] = {p}: must be finite and in [-1.0, 1.0]"),
                });
            }
        }
        let mut config = Self::with_gains(gains)?;
        config.pans = Some(pans);
        Ok(config)
    }

    /// Set whether to clip the mixed output to [-1.0, 1.0].
    #[must_use]
    pub fn with_clip(mut self, clip: bool) -> Self {
        self.clip_output = clip;
        self
    }

    /// Set per-voice pitch shift in semitones.
    ///
    /// Even small values (±0.05 to ±0.10 = ±5-10 cents) create a richer sound.
    /// Length must equal `n_voices`. Values must be in [-12.0, 12.0].
    #[must_use]
    pub fn with_pitch_semitones(mut self, semitones: Vec<f32>) -> Self {
        self.pitch_semitones = Some(semitones);
        self
    }

    /// Set per-voice timing offsets in seconds.
    ///
    /// Small offsets (±0.002 to ±0.005 = ±2-5ms) break up robotic unison.
    /// Length must equal `n_voices`. Values must be in [-0.050, 0.050].
    #[must_use]
    pub fn with_timing_offsets(mut self, offsets_sec: Vec<f32>) -> Self {
        self.timing_offsets_sec = Some(offsets_sec);
        self
    }

    /// Set stereo width (0.0 = mono, 1.0 = full stereo).
    ///
    /// Scales all pan positions: `effective_pan = pan * stereo_width`.
    #[must_use]
    pub fn with_stereo_width(mut self, width: f32) -> Self {
        self.stereo_width = width;
        self
    }

    /// Enable soft tanh limiter instead of hard clipping.
    ///
    /// `drive` controls saturation onset. Typical: 1.0 (gentle) to 2.0 (more compressed).
    /// Formula: `tanh(x * drive) / drive`.
    #[must_use]
    pub fn with_soft_limiter(mut self, drive: f32) -> Self {
        self.soft_limiter_drive = Some(drive);
        self
    }

    /// Enable `1/sqrt(n_voices)` gain normalization for better perceived loudness.
    #[must_use]
    pub fn with_sqrt_gain_normalization(mut self) -> Self {
        self.sqrt_gain_normalization = true;
        self
    }

    /// Enable spatial reverb with default settings.
    ///
    /// Applies a Schroeder reverb (4 comb + 2 allpass filters) with early
    /// reflections for choir-like spatial depth. Default: reverb_mix=0.15,
    /// room_size=0.3, damping=0.5, early_reflections=true.
    #[must_use]
    pub fn with_reverb(mut self, config: ReverbConfig) -> Self {
        self.reverb = Some(config);
        self
    }

    /// Enable spatial reverb with default parameters.
    ///
    /// Convenience method equivalent to `with_reverb(ReverbConfig::default())`.
    #[must_use]
    pub fn with_default_reverb(self) -> Self {
        self.with_reverb(ReverbConfig::default())
    }

    /// Build a [`StereoChorusConfig`] from this chorus config.
    ///
    /// If `pans` are set, converts them to [`StereoPosition::Custom`] values.
    /// Otherwise uses [`default_voice_layout`] for automatic LCR spread.
    /// Stereo width and mono compatibility are carried over.
    pub fn to_stereo_config(&self) -> StereoChorusConfig {
        let positions = if let Some(ref pans) = self.pans {
            pans.iter()
                .map(|&p| StereoPosition::from_pan(p * self.stereo_width))
                .collect()
        } else {
            default_voice_layout(self.n_voices)
        };
        StereoChorusConfig {
            positions,
            stereo_width: self.stereo_width,
            mono_compatible: false,
        }
    }

    /// Set per-voice stereo positions using [`StereoPosition`] values.
    ///
    /// Converts positions to pan values and stores them in `pans`.
    /// This is a convenience wrapper for [`with_stereo_pan`] that
    /// accepts the higher-level position enum.
    #[must_use]
    pub fn with_stereo_positions(mut self, positions: Vec<StereoPosition>) -> Self {
        let pans: Vec<f32> = positions.iter().map(|p| p.to_pan()).collect();
        self.pans = Some(pans);
        self
    }

    /// Create a chorus config with equal-power (sqrt) gain normalization.
    ///
    /// Each voice gets gain = `1.0 / sqrt(n_voices)`, preserving perceived
    /// loudness better than `1/n_voices` for uncorrelated signals.
    pub fn equal_power(n_voices: usize) -> Result<Self, KokoroError> {
        if n_voices == 0 || n_voices > 32 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: format!("must be 1..=32, got {n_voices}"),
            });
        }
        let gain = 1.0 / (n_voices as f32).sqrt();
        Ok(Self {
            n_voices,
            gains: vec![gain; n_voices],
            clip_output: true,
            pans: None,
            pitch_semitones: None,
            timing_offsets_sec: None,
            stereo_width: 1.0,
            soft_limiter_drive: None,
            sqrt_gain_normalization: true,
            reverb: None,
        })
    }

    /// Create a "rich chorus" preset: sqrt gains, soft limiter, slight detuning, reverb.
    ///
    /// Applies sensible defaults for natural-sounding multi-voice output:
    /// - `1/sqrt(N)` gain normalization for full loudness
    /// - ±8 cent detuning spread across voices for richness
    /// - ±3ms timing offsets for natural attack variation
    /// - Soft tanh limiter at drive=1.5 for smooth peak handling
    /// - Spatial reverb (mix=0.15, room_size=0.3) for choir-like depth
    pub fn rich_chorus(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::equal_power(n_voices)?;
        config.soft_limiter_drive = Some(1.5);

        // Spread pitch ±8 cents (0.08 semitones) symmetrically.
        let max_detune: f32 = 0.08;
        let pitches: Vec<f32> = (0..n_voices)
            .map(|i| {
                if n_voices == 1 {
                    0.0
                } else {
                    let t = i as f32 / (n_voices - 1) as f32; // 0..1
                    -max_detune + 2.0 * max_detune * t
                }
            })
            .collect();
        config.pitch_semitones = Some(pitches);

        // Spread timing ±3ms symmetrically.
        let max_offset: f32 = 0.003;
        let offsets: Vec<f32> = (0..n_voices)
            .map(|i| {
                if n_voices == 1 {
                    0.0
                } else {
                    let t = i as f32 / (n_voices - 1) as f32;
                    -max_offset + 2.0 * max_offset * t
                }
            })
            .collect();
        config.timing_offsets_sec = Some(offsets);

        // Add spatial reverb for choir-like depth.
        config.reverb = Some(ReverbConfig::default());

        // Auto-spread voices across stereo field (LCR layout).
        let layout = default_voice_layout(n_voices);
        let pans: Vec<f32> = layout.iter().map(|p| p.to_pan()).collect();
        config.pans = Some(pans);

        Ok(config)
    }

    /// Validate that the config is internally consistent.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.n_voices == 0 || self.n_voices > 32 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: format!("must be 1..=32, got {}", self.n_voices),
            });
        }
        if self.gains.len() != self.n_voices {
            return Err(KokoroError::InvalidConfig {
                field: "gains",
                reason: format!(
                    "gains length {} != n_voices {}",
                    self.gains.len(),
                    self.n_voices,
                ),
            });
        }
        if let Some(ref pans) = self.pans {
            if pans.len() != self.n_voices {
                return Err(KokoroError::InvalidConfig {
                    field: "pans",
                    reason: format!("pans length {} != n_voices {}", pans.len(), self.n_voices),
                });
            }
        }
        if let Some(ref pitches) = self.pitch_semitones {
            if pitches.len() != self.n_voices {
                return Err(KokoroError::InvalidConfig {
                    field: "pitch_semitones",
                    reason: format!(
                        "pitch_semitones length {} != n_voices {}",
                        pitches.len(),
                        self.n_voices,
                    ),
                });
            }
            for (i, &p) in pitches.iter().enumerate() {
                if !p.is_finite() || !(-12.0..=12.0).contains(&p) {
                    return Err(KokoroError::InvalidConfig {
                        field: "pitch_semitones",
                        reason: format!(
                            "pitch_semitones[{i}] = {p}: must be finite and in [-12.0, 12.0]"
                        ),
                    });
                }
            }
        }
        if let Some(ref offsets) = self.timing_offsets_sec {
            if offsets.len() != self.n_voices {
                return Err(KokoroError::InvalidConfig {
                    field: "timing_offsets_sec",
                    reason: format!(
                        "timing_offsets_sec length {} != n_voices {}",
                        offsets.len(),
                        self.n_voices,
                    ),
                });
            }
            for (i, &t) in offsets.iter().enumerate() {
                if !t.is_finite() || !(-0.050..=0.050).contains(&t) {
                    return Err(KokoroError::InvalidConfig {
                        field: "timing_offsets_sec",
                        reason: format!(
                            "timing_offsets_sec[{i}] = {t}: must be finite and in [-0.050, 0.050]"
                        ),
                    });
                }
            }
        }
        if !self.stereo_width.is_finite() || !(0.0..=1.0).contains(&self.stereo_width) {
            return Err(KokoroError::InvalidConfig {
                field: "stereo_width",
                reason: format!(
                    "stereo_width = {}: must be finite and in [0.0, 1.0]",
                    self.stereo_width,
                ),
            });
        }
        if let Some(drive) = self.soft_limiter_drive {
            if !drive.is_finite() || drive <= 0.0 || drive > 10.0 {
                return Err(KokoroError::InvalidConfig {
                    field: "soft_limiter_drive",
                    reason: format!(
                        "soft_limiter_drive = {drive}: must be finite and in (0.0, 10.0]"
                    ),
                });
            }
        }
        if let Some(ref reverb) = self.reverb {
            reverb.validate()?;
        }
        Ok(())
    }

    /// Duration of mixed audio in seconds given the longest voice's sample count.
    #[must_use]
    pub fn duration_secs(&self, max_samples: usize) -> f64 {
        max_samples as f64 / KOKORO_SAMPLE_RATE as f64
    }

    /// Return per-voice F0 scale factors from [`pitch_semitones`](Self::pitch_semitones).
    ///
    /// Each factor is `2^(semitones/12)`. Returns `None` if `pitch_semitones`
    /// is not set. Useful for callers that want to scale F0 at synthesis time
    /// rather than relying on post-synthesis resampling.
    #[must_use]
    pub fn pitch_factors(&self) -> Option<Vec<f32>> {
        self.pitch_semitones
            .as_ref()
            .map(|semitones| semitones.iter().map(|&s| pitch_shift_factor(s)).collect())
    }

    /// Recompute gains based on the current `sqrt_gain_normalization` flag.
    ///
    /// When `sqrt_gain_normalization` is true, sets all gains to `1/sqrt(n_voices)`.
    /// When false, sets all gains to `1/n_voices`. Useful after toggling the
    /// flag via [`with_sqrt_gain_normalization`](Self::with_sqrt_gain_normalization).
    #[must_use]
    pub fn with_normalized_gains(mut self) -> Self {
        let n = self.n_voices as f32;
        let gain = if self.sqrt_gain_normalization {
            1.0 / n.sqrt()
        } else {
            1.0 / n
        };
        self.gains = vec![gain; self.n_voices];
        self
    }
}

// ---------------------------------------------------------------------------
// Per-voice input
// ---------------------------------------------------------------------------

/// Input for a single voice in a chorus synthesis call.
///
/// Each voice has its own phoneme token IDs, style embedding, and speed.
/// The synthesis backend dispatches these independently (with shared weights).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VoiceInput {
    /// Token IDs for this voice's text. Shape: `[1, T]` where T is the
    /// sequence length (including padding tokens at start and end).
    pub token_ids: Vec<u32>,

    /// Style embedding index or vector for this voice.
    /// In the simplest case, all voices share the same style.
    /// For mixed chorus, each voice gets a different speaker style.
    pub style_index: usize,

    /// Speaking rate multiplier for this voice (1.0 = normal).
    /// Varying speed across voices creates a richer ensemble effect.
    pub speed: f32,
}

impl VoiceInput {
    /// Create a voice input.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidSpeed` if `speed` is non-finite, zero, or negative.
    pub fn new(token_ids: Vec<u32>, style_index: usize, speed: f32) -> Result<Self, KokoroError> {
        validate_speed(speed)?;
        Ok(Self {
            token_ids,
            style_index,
            speed,
        })
    }
}

// ---------------------------------------------------------------------------
// Stereo mixing parameters
// ---------------------------------------------------------------------------

/// Per-voice mixing parameters for stereo output.
///
/// Used by [`mix_voices_stereo`] to position each voice in the stereo field
/// using equal-power pan law.
#[derive(Debug, Clone, Copy)]
pub struct VoiceMix {
    /// Gain multiplier, must be in [0.0, 1.0].
    pub gain: f32,
    /// Stereo pan position: -1.0 = hard left, 0.0 = center, 1.0 = hard right.
    pub pan: f32,
}

// ---------------------------------------------------------------------------
// Soft limiter
// ---------------------------------------------------------------------------

/// Apply soft tanh limiter: `tanh(x * drive) / drive`.
///
/// Smoothly saturates peaks instead of hard clipping at ±1.0.
/// At `drive = 1.0` the function is gentle (tanh is nearly linear near 0).
/// At higher drive values, saturation engages earlier for more compression.
///
/// Properties: output is always in `(-1/drive, 1/drive)` which for `drive >= 1.0`
/// keeps output in `(-1, 1)`. The function is odd-symmetric and monotonic.
#[inline]
fn soft_limit(sample: f32, drive: f32) -> f32 {
    (sample * drive).tanh() / drive
}

/// Apply soft limiting to all samples in a buffer.
fn apply_soft_limiter(buffer: &mut [f32], drive: f32) {
    for sample in buffer.iter_mut() {
        *sample = soft_limit(*sample, drive);
    }
}

/// Apply timing offset to a PCM buffer by shifting samples.
///
/// Positive offset delays the signal (prepends silence, truncates end).
/// Negative offset advances the signal (skips beginning, appends silence).
/// Returns a new buffer of the same length as `target_len`.
fn apply_timing_offset(pcm: &[f32], offset_samples: isize, target_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; target_len];
    if offset_samples >= 0 {
        let skip = offset_samples as usize;
        for (i, &s) in pcm.iter().enumerate() {
            let dst = i + skip;
            if dst < target_len {
                out[dst] = s;
            }
        }
    } else {
        let skip = (-offset_samples) as usize;
        for (i, &s) in pcm.iter().enumerate() {
            if i >= skip {
                let dst = i - skip;
                if dst < target_len {
                    out[dst] = s;
                }
            }
        }
    }
    out
}

/// Compute the F0 scale factor for a pitch shift in semitones.
///
/// Returns `2^(semitones / 12)`. At 0 semitones returns 1.0 (no change).
/// At +12 returns 2.0 (octave up), at -12 returns 0.5 (octave down).
#[inline]
#[must_use]
pub fn pitch_shift_factor(semitones: f32) -> f32 {
    (2.0f32).powf(semitones / 12.0)
}

/// Apply pitch shift to PCM audio via linear-interpolation resampling.
///
/// Shifts pitch by the factor `2^(semitones/12)`. Positive semitones raise
/// pitch (shorter output), negative semitones lower pitch (longer output).
/// The output is truncated or zero-padded to `target_len` samples.
///
/// For small chorus detuning (±5-10 cents), linear interpolation introduces
/// negligible artifacts. For larger shifts (>1 semitone), higher-order
/// interpolation would be preferred.
fn apply_pitch_shift(pcm: &[f32], semitones: f32, target_len: usize) -> Vec<f32> {
    if semitones.abs() < 1e-6 || pcm.is_empty() {
        // No shift needed — copy and pad/truncate to target_len.
        let mut out = vec![0.0f32; target_len];
        let copy_len = pcm.len().min(target_len);
        out[..copy_len].copy_from_slice(&pcm[..copy_len]);
        return out;
    }

    let rate = pitch_shift_factor(semitones);
    let mut out = vec![0.0f32; target_len];

    for (i, sample) in out.iter_mut().enumerate() {
        // Map output sample index to input position via the pitch rate.
        let src_pos = i as f64 * f64::from(rate);
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        if src_idx >= pcm.len() {
            break; // Past the end of input — remaining output stays zero.
        }

        let s0 = pcm[src_idx];
        let s1 = if src_idx + 1 < pcm.len() {
            pcm[src_idx + 1]
        } else {
            s0 // Repeat last sample at boundary.
        };
        *sample = s0 + frac * (s1 - s0);
    }

    out
}

// ---------------------------------------------------------------------------
// Audio mixing
// ---------------------------------------------------------------------------

/// Mix multiple voice audio buffers into a single output.
///
/// Each voice's PCM is multiplied by its gain, then all voices are summed
/// sample-by-sample. The output length equals the longest voice. Shorter
/// voices are zero-padded (silence after their audio ends).
///
/// # Arguments
///
/// * `voice_audio` - PCM audio for each voice (24kHz mono, [-1.0, 1.0]).
/// * `gains` - Per-voice gain multiplier. Length must match `voice_audio`.
/// * `clip` - If true, clamp output to [-1.0, 1.0].
///
/// # Errors
///
/// Returns `KokoroError::InvalidInput` if `voice_audio` and `gains` lengths
/// differ, or if any gain is non-finite.
pub fn mix_voices(
    voice_audio: &[Vec<f32>],
    gains: &[f32],
    clip: bool,
) -> Result<Vec<f32>, KokoroError> {
    if voice_audio.len() != gains.len() {
        return Err(KokoroError::InvalidInput(format!(
            "voice_audio length {} != gains length {}",
            voice_audio.len(),
            gains.len(),
        )));
    }
    if voice_audio.is_empty() {
        return Ok(Vec::new());
    }

    // Output length = longest voice.
    let max_len = voice_audio.iter().map(Vec::len).max().unwrap_or(0);
    if max_len == 0 {
        return Ok(Vec::new());
    }

    let mut mixed = vec![0.0f32; max_len];

    for (voice_pcm, &gain) in voice_audio.iter().zip(gains.iter()) {
        let g = gain.clamp(0.0, 1.0);
        for (i, &sample) in voice_pcm.iter().enumerate() {
            mixed[i] += sample * g;
        }
    }

    if clip {
        for sample in &mut mixed {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    Ok(mixed)
}

/// Mix borrowed voice slices into mono output (ref-slice variant of [`mix_voices`]).
fn mix_voices_from_ref_slices(
    voice_audio: &[&[f32]],
    gains: &[f32],
    clip: bool,
) -> Result<Vec<f32>, KokoroError> {
    if voice_audio.len() != gains.len() {
        return Err(KokoroError::InvalidInput(format!(
            "voice_audio length {} != gains length {}",
            voice_audio.len(),
            gains.len(),
        )));
    }
    if voice_audio.is_empty() {
        return Ok(Vec::new());
    }
    let max_len = voice_audio.iter().map(|v| v.len()).max().unwrap_or(0);
    if max_len == 0 {
        return Ok(Vec::new());
    }
    let mut mixed = vec![0.0f32; max_len];
    for (pcm, &gain) in voice_audio.iter().zip(gains.iter()) {
        let g = gain.clamp(0.0, 1.0);
        for (i, &sample) in pcm.iter().enumerate() {
            mixed[i] += sample * g;
        }
    }
    if clip {
        for sample in &mut mixed {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }
    Ok(mixed)
}

/// Mix borrowed voice slices into stereo output (ref-slice variant of [`mix_voices_stereo`]).
fn mix_voices_stereo_from_refs(
    voice_audio: &[&[f32]],
    mix_params: &[VoiceMix],
    clip: bool,
) -> Result<Vec<f32>, KokoroError> {
    if voice_audio.len() != mix_params.len() {
        return Err(KokoroError::InvalidInput(format!(
            "voice_audio length {} != mix_params length {}",
            voice_audio.len(),
            mix_params.len(),
        )));
    }
    if voice_audio.is_empty() {
        return Ok(Vec::new());
    }
    let max_len = voice_audio.iter().map(|v| v.len()).max().unwrap_or(0);
    if max_len == 0 {
        return Ok(Vec::new());
    }
    let mut stereo = vec![0.0f32; max_len * 2];
    for (pcm, params) in voice_audio.iter().zip(mix_params) {
        let g = params.gain.clamp(0.0, 1.0);
        let angle = ((params.pan + 1.0) * 0.5).clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
        let left_gain = angle.cos() * g;
        let right_gain = angle.sin() * g;
        for (i, &sample) in pcm.iter().enumerate() {
            stereo[i * 2] += sample * left_gain;
            stereo[i * 2 + 1] += sample * right_gain;
        }
    }
    if clip {
        for sample in &mut stereo {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }
    Ok(stereo)
}

/// Mix voice audio slices into a single output using a [`ChorusConfig`].
///
/// Like [`mix_voices_with_config`] but accepts borrowed slices, avoiding
/// the need to clone PCM data into `Vec<f32>` intermediaries. Used by
/// [`assemble_streaming_chorus`](crate::kokoro_streaming::assemble_streaming_chorus)
/// to eliminate per-chunk clone overhead.
///
/// Applies pitch shifting, timing offsets, stereo width, and soft limiting
/// when configured.
pub fn mix_voices_from_refs(
    voice_audio: &[&[f32]],
    config: &ChorusConfig,
) -> Result<Vec<f32>, KokoroError> {
    config.validate()?;
    if voice_audio.len() != config.n_voices {
        return Err(KokoroError::InvalidInput(format!(
            "voice_audio length {} != config.n_voices {}",
            voice_audio.len(),
            config.n_voices,
        )));
    }

    // Apply pitch shifting and/or timing offsets if configured.
    // Both require allocating new buffers for transformed voices.
    let has_pitch = config.pitch_semitones.is_some();
    let has_offsets = config.timing_offsets_sec.is_some();
    let max_len = voice_audio.iter().map(|v| v.len()).max().unwrap_or(0);

    if has_pitch || has_offsets {
        // Build per-voice transformed buffers. Pitch shift first (changes
        // content timing), then timing offset (shifts position in the mix).
        let pitches = config.pitch_semitones.as_deref();
        let offsets = config.timing_offsets_sec.as_deref();

        let transformed: Vec<Vec<f32>> = voice_audio
            .iter()
            .enumerate()
            .map(|(i, pcm)| {
                // Step 1: pitch shift via resampling.
                let pitched: std::borrow::Cow<'_, [f32]> =
                    if let Some(semitones) = pitches.and_then(|p| p.get(i)).copied() {
                        if semitones.abs() > 1e-6 {
                            std::borrow::Cow::Owned(apply_pitch_shift(pcm, semitones, max_len))
                        } else {
                            std::borrow::Cow::Borrowed(*pcm)
                        }
                    } else {
                        std::borrow::Cow::Borrowed(*pcm)
                    };

                // Step 2: timing offset.
                if let Some(&offset_sec) = offsets.and_then(|o| o.get(i)) {
                    let offset_samples = (offset_sec * KOKORO_SAMPLE_RATE as f32).round() as isize;
                    if offset_samples != 0 {
                        return apply_timing_offset(&pitched, offset_samples, max_len);
                    }
                }

                pitched.into_owned()
            })
            .collect();

        let refs: Vec<&[f32]> = transformed.iter().map(Vec::as_slice).collect();
        mix_voices_from_refs_inner(&refs, config)
    } else {
        mix_voices_from_refs_inner(voice_audio, config)
    }
}

/// Inner mixing logic for ref-slice paths (after timing offsets applied).
fn mix_voices_from_refs_inner(
    voice_audio: &[&[f32]],
    config: &ChorusConfig,
) -> Result<Vec<f32>, KokoroError> {
    let use_soft_limiter = config.soft_limiter_drive.is_some();
    // When soft limiter is active, don't hard-clip during mixing (limiter handles it).
    let clip = config.clip_output && !use_soft_limiter;

    let is_stereo = config.pans.is_some();

    let mut mixed = if let Some(ref pans) = config.pans {
        let width = config.stereo_width;
        let mix_params: Vec<VoiceMix> = config
            .gains
            .iter()
            .zip(pans.iter())
            .map(|(&gain, &pan)| VoiceMix {
                gain,
                pan: pan * width,
            })
            .collect();
        mix_voices_stereo_from_refs(voice_audio, &mix_params, clip)?
    } else {
        mix_voices_from_ref_slices(voice_audio, &config.gains, clip)?
    };

    if let Some(drive) = config.soft_limiter_drive {
        apply_soft_limiter(&mut mixed, drive);
    }

    // Apply spatial reverb if configured.
    if let Some(ref reverb_config) = config.reverb {
        let effective_pans: Option<Vec<f32>> = config
            .pans
            .as_ref()
            .map(|pans| pans.iter().map(|&p| p * config.stereo_width).collect());
        apply_reverb(
            &mut mixed,
            reverb_config,
            is_stereo,
            Some(voice_audio),
            effective_pans.as_deref(),
            Some(&config.gains),
        )?;
    }

    Ok(mixed)
}

/// Mix voices using a [`ChorusConfig`] for convenience.
///
/// When the config has stereo pans set, produces interleaved stereo output
/// via [`mix_voices_stereo`]. Otherwise produces mono via [`mix_voices`].
///
/// Applies pitch shifting, timing offsets, stereo width, and soft limiting
/// when configured.
pub fn mix_voices_with_config(
    voice_audio: &[Vec<f32>],
    config: &ChorusConfig,
) -> Result<Vec<f32>, KokoroError> {
    config.validate()?;
    if voice_audio.len() != config.n_voices {
        return Err(KokoroError::InvalidInput(format!(
            "voice_audio length {} != config.n_voices {}",
            voice_audio.len(),
            config.n_voices,
        )));
    }

    // Delegate to ref-slice path to share timing-offset/soft-limiter logic.
    let refs: Vec<&[f32]> = voice_audio.iter().map(Vec::as_slice).collect();
    mix_voices_from_refs(&refs, config)
}

/// Mix multiple voice PCM buffers to interleaved stereo using equal-power pan.
///
/// Output format: `[L0, R0, L1, R1, ...]` (interleaved stereo at 24kHz).
///
/// Uses equal-power pan law:
/// - `angle = (pan + 1) * 0.5 * π/2`  (maps [-1, 1] → [0, π/2])
/// - `left_gain  = cos(angle) * gain`
/// - `right_gain = sin(angle) * gain`
///
/// At center pan (0.0): left = right ≈ 0.707 * gain (equal power).
///
/// This matches dvoice's `mix_choir_parts_stereo()` algorithm.
///
/// # Errors
///
/// Returns `KokoroError::InvalidInput` if `voice_audio` and `mix_params`
/// lengths differ.
pub fn mix_voices_stereo(
    voice_audio: &[Vec<f32>],
    mix_params: &[VoiceMix],
    clip: bool,
) -> Result<Vec<f32>, KokoroError> {
    if voice_audio.len() != mix_params.len() {
        return Err(KokoroError::InvalidInput(format!(
            "voice_audio length {} != mix_params length {}",
            voice_audio.len(),
            mix_params.len(),
        )));
    }
    if voice_audio.is_empty() {
        return Ok(Vec::new());
    }

    let max_len = voice_audio.iter().map(Vec::len).max().unwrap_or(0);
    if max_len == 0 {
        return Ok(Vec::new());
    }

    let mut stereo = vec![0.0f32; max_len * 2];

    for (pcm, params) in voice_audio.iter().zip(mix_params) {
        let g = params.gain.clamp(0.0, 1.0);
        let angle = ((params.pan + 1.0) * 0.5).clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
        let left_gain = angle.cos() * g;
        let right_gain = angle.sin() * g;

        for (i, &sample) in pcm.iter().enumerate() {
            stereo[i * 2] += sample * left_gain;
            stereo[i * 2 + 1] += sample * right_gain;
        }
    }

    if clip {
        for sample in &mut stereo {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    Ok(stereo)
}

#[cfg(kani)]
#[path = "kokoro_chorus_kani_tests.rs"]
mod kani_proofs;

#[cfg(test)]
#[path = "kokoro_chorus_tests.rs"]
mod tests;
