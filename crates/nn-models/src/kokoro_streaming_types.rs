// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Data types for streaming Kokoro synthesis.
//!
//! Extracted from `kokoro_streaming.rs` (#3351) to keep the main file
//! under the 500-line limit. Re-exported via `kokoro_streaming`.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Crossfade window shape
// ---------------------------------------------------------------------------

/// Window function used for crossfade blending between adjacent chunks.
///
/// Controls the interpolation curve in the overlap region.
///
/// - **Linear**: `alpha = i / (N-1)`. Simple, suitable for short overlaps
///   (< 40ms / 960 samples at 24kHz).
/// - **Hann**: `alpha = 0.5 * (1 - cos(pi * i / (N-1)))`. Smoother
///   energy transition, power-complementary (sum of squares = 1).
/// - **SqrtHann**: `alpha = sqrt(0.5 * (1 - cos(pi * i / (N-1))))`.
///   Amplitude-complementary (`(1-alpha) + alpha = 1`), which preserves
///   perceived loudness across the crossfade. Best for TTS/speech
///   streaming where energy dips are audible. Default for chorus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum CrossfadeWindow {
    /// Linear interpolation: `alpha = i / (N-1)`.
    ///
    /// Matches dvoice's `sentence.rs` crossfade and
    /// `streaming.rs::crossfade_linear`. Preferred for short overlaps.
    Linear,

    /// Hann (raised cosine) window: `alpha = 0.5 * (1 - cos(pi * i / (N-1)))`.
    ///
    /// Produces smoother energy transitions at chunk boundaries compared
    /// to linear crossfade. Power-complementary (sum of squares = 1).
    Hann,

    /// Sqrt-Hann (root raised cosine) window:
    /// `alpha = sqrt(0.5 * (1 - cos(pi * i / (N-1))))`.
    ///
    /// Amplitude-complementary: the fade-out weight `(1-alpha)` plus the
    /// fade-in weight `alpha` sum to 1.0 at every sample, preserving
    /// perceived loudness throughout the crossfade region. Regular Hann
    /// only preserves power (sum of squares = 1), which can produce
    /// audible energy dips for speech signals.
    ///
    /// Preferred for streaming TTS chorus where constant perceived
    /// loudness matters more than spectral smoothness.
    #[default]
    SqrtHann,
}

/// Threshold in samples above which Hann window is used by default.
///
/// 960 samples = 40ms at 24kHz. Below this, linear crossfade is adequate;
/// above, the Hann window provides better spectral continuity.
pub const HANN_CROSSFADE_THRESHOLD: usize = 960;


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for streaming Kokoro synthesis.
///
/// Controls crossfade behavior at chunk boundaries. The default is 40ms
/// (960 samples) at 24kHz with a Hann window — enough to smooth F0
/// discontinuities with good spectral continuity.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct KokoroStreamConfig {
    /// Crossfade overlap in samples between adjacent chunks.
    ///
    /// Default: 960 (40ms at 24kHz). Must be > 0.
    pub crossfade_samples: usize,

    /// Window function for crossfade blending.
    ///
    /// Default: [`CrossfadeWindow::SqrtHann`]. Uses sqrt-Hann (root raised
    /// cosine) for the default 40ms overlap -- amplitude-complementary,
    /// preserving perceived loudness for TTS/speech. Linear is still
    /// available for short overlaps, Hann for backward compatibility.
    pub crossfade_window: CrossfadeWindow,
}

impl KokoroStreamConfig {
    /// Create a config with the given crossfade length.
    ///
    /// Automatically selects the crossfade window based on overlap duration:
    /// - SqrtHann window for `crossfade_samples >= 960` (40ms at 24kHz) --
    ///   amplitude-complementary, best for TTS/speech streaming
    /// - Linear window for shorter overlaps
    ///
    /// Use [`with_window`](Self::with_window) to override the automatic selection.
    pub fn new(crossfade_samples: usize) -> Result<Self, KokoroError> {
        let crossfade_window = if crossfade_samples >= HANN_CROSSFADE_THRESHOLD {
            CrossfadeWindow::SqrtHann
        } else {
            CrossfadeWindow::Linear
        };
        let config = Self {
            crossfade_samples,
            crossfade_window,
        };
        config.validate()?;
        Ok(config)
    }

    /// Set the crossfade window function.
    ///
    /// Returns `self` for builder-style chaining.
    #[must_use]
    pub fn with_window(mut self, window: CrossfadeWindow) -> Self {
        self.crossfade_window = window;
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.crossfade_samples == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "crossfade_samples",
                reason: "must be > 0".into(),
            });
        }
        Ok(())
    }

    /// Crossfade duration in seconds.
    #[must_use]
    pub fn crossfade_duration_secs(&self) -> f64 {
        self.crossfade_samples as f64 / KOKORO_SAMPLE_RATE as f64
    }
}

impl Default for KokoroStreamConfig {
    fn default() -> Self {
        Self {
            crossfade_samples: 960, // 40ms at 24kHz
            crossfade_window: CrossfadeWindow::SqrtHann,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio chunk
// ---------------------------------------------------------------------------

/// A chunk of synthesized PCM audio from streaming synthesis.
///
/// Each chunk contains ready-to-play audio with crossfade already applied
/// at boundaries. The consumer (dvoice conductor) can play chunks
/// sequentially without additional processing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AudioChunk {
    /// Raw PCM samples at 24kHz, normalized to [-1.0, 1.0].
    ///
    /// For mono (`channels == 1`): sequential samples.
    /// For stereo (`channels == 2`): interleaved `[L0, R0, L1, R1, ...]`.
    ///
    /// For the first chunk, this is the full synthesis output (minus the
    /// crossfade tail reserved for blending with the next chunk).
    /// For subsequent chunks, the leading `crossfade_samples` are blended
    /// with the previous chunk's tail.
    pub pcm: Vec<f32>,

    /// Number of audio channels (1 = mono, 2 = interleaved stereo).
    pub channels: usize,

    /// Sample offset in the full concatenated output stream.
    ///
    /// Chunk 0 starts at offset 0. Each subsequent chunk's offset accounts
    /// for the crossfade overlap with the previous chunk.
    pub sample_offset: usize,

    /// Zero-based index of this chunk in the sequence.
    pub chunk_index: usize,

    /// Total number of chunks in the sequence.
    pub total_chunks: usize,

    /// Whether this is the last chunk in the sequence.
    pub is_final: bool,
}

impl AudioChunk {
    /// Create a new audio chunk.
    ///
    /// Prefer [`assemble_streaming_chunks`](super::assemble_streaming_chunks)
    /// or [`assemble_streaming_chorus`](super::assemble_streaming_chorus)
    /// over constructing directly.
    #[must_use]
    pub fn new(
        pcm: Vec<f32>,
        channels: usize,
        sample_offset: usize,
        chunk_index: usize,
        total_chunks: usize,
        is_final: bool,
    ) -> Self {
        Self {
            pcm,
            channels,
            sample_offset,
            chunk_index,
            total_chunks,
            is_final,
        }
    }

    /// Duration of this chunk in seconds at 24kHz.
    ///
    /// For stereo, divides by channels to get time (not raw float count).
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        let ch = self.channels.max(1);
        self.pcm.len() as f64 / (KOKORO_SAMPLE_RATE as f64 * ch as f64)
    }

    /// Number of PCM samples in this chunk.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pcm.len()
    }

    /// Whether this chunk contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pcm.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Concatenate (utility for consumers that want a single buffer)
// ---------------------------------------------------------------------------

/// Concatenate a sequence of [`AudioChunk`]s into a single PCM buffer.
///
/// Useful for testing or when the consumer wants the full waveform rather
/// than incremental chunks.
#[must_use]
pub fn concatenate_chunks(chunks: &[AudioChunk]) -> Vec<f32> {
    let total: usize = chunks.iter().map(|c| c.pcm.len()).sum();
    let mut out = Vec::with_capacity(total);
    for chunk in chunks {
        out.extend_from_slice(&chunk.pcm);
    }
    out
}
