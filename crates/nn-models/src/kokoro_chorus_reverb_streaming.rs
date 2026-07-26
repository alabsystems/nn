// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming reverb processor and preset configurations.
//!
//! In streaming synthesis, audio is produced in chunks. A plain
//! [`StereoReverb`] created per-chunk loses its delay line state at each
//! chunk boundary, causing audible discontinuities in the reverb tail.
//! [`StreamingReverb`] wraps a persistent `StereoReverb` and retains it
//! across calls to [`process_chunk`](StreamingReverb::process_chunk),
//! ensuring the reverb tail from one chunk feeds naturally into the next.
//!
//! Also provides convenient preset constructors on [`ReverbConfig`] for
//! common room types (small room, medium hall, large church, cathedral).

use super::kokoro_chorus_reverb::{ReverbConfig, StereoReverb};
use crate::kokoro_error::KokoroError;
use crate::kokoro_streaming::AudioChunk;

// ---------------------------------------------------------------------------
// ReverbConfig presets
// ---------------------------------------------------------------------------

impl ReverbConfig {
    /// Small room preset: subtle, close-sounding reverb.
    ///
    /// Good for spoken word or when you want just a hint of space.
    /// RT60 ~0.2s, low mix, bright character.
    #[must_use]
    pub fn small_room() -> Self {
        Self {
            reverb_mix: 0.10,
            room_size: 0.15,
            early_reflections: true,
            damping: 0.3,
        }
    }

    /// Medium hall preset: moderate reverb for choral performances.
    ///
    /// Natural-sounding space that adds depth without overwhelming voices.
    /// RT60 ~0.8s, moderate mix. The default preset for chorus work.
    #[must_use]
    pub fn medium_hall() -> Self {
        Self {
            reverb_mix: 0.20,
            room_size: 0.45,
            early_reflections: true,
            damping: 0.5,
        }
    }

    /// Large church preset: spacious, warm reverb.
    ///
    /// Strong sense of space with pronounced tail. Works well for
    /// slow, sustained choir passages. RT60 ~1.5s.
    #[must_use]
    pub fn large_church() -> Self {
        Self {
            reverb_mix: 0.30,
            room_size: 0.70,
            early_reflections: true,
            damping: 0.6,
        }
    }

    /// Cathedral preset: very long, diffuse reverb tail.
    ///
    /// Dramatic, wash-like reverb for maximum spatial effect. RT60 ~2s+.
    /// Best used sparingly -- high mix values can muddy fast speech.
    #[must_use]
    pub fn cathedral() -> Self {
        Self {
            reverb_mix: 0.35,
            room_size: 0.90,
            early_reflections: true,
            damping: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming reverb (state persists across chunks)
// ---------------------------------------------------------------------------

/// Streaming-aware reverb processor that persists state across audio chunks.
///
/// In streaming synthesis, audio is produced in chunks. A plain
/// [`StereoReverb`] created per-chunk would lose its delay line contents
/// (comb and allpass filter state) at each chunk boundary, causing audible
/// discontinuities in the reverb tail. `StreamingReverb` wraps a
/// `StereoReverb` and retains it across calls to
/// [`process_chunk`](Self::process_chunk), ensuring the reverb tail from
/// one chunk feeds naturally into the next.
///
/// # Usage
///
/// ```text
/// let mut reverb = StreamingReverb::new(config, is_stereo)?;
/// for chunk in audio_chunks {
///     reverb.process_chunk(&mut chunk.pcm);
/// }
/// ```
pub struct StreamingReverb {
    inner: StereoReverb,
    config: ReverbConfig,
    is_stereo: bool,
}

impl StreamingReverb {
    /// Create a new streaming reverb processor.
    ///
    /// # Arguments
    ///
    /// * `config` - Reverb parameters (room size, mix, damping, etc.).
    /// * `is_stereo` - Whether audio buffers are interleaved stereo.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails.
    pub fn new(config: ReverbConfig, is_stereo: bool) -> Result<Self, KokoroError> {
        config.validate()?;
        let inner = StereoReverb::new(&config);
        Ok(Self {
            inner,
            config,
            is_stereo,
        })
    }

    /// Process one audio chunk in-place, preserving reverb state for the
    /// next chunk.
    ///
    /// Applies late Schroeder reverb. Early reflections are NOT applied
    /// here because they require per-voice audio data which is not
    /// available at the post-mix stage. Early reflections are applied
    /// during voice mixing (in `mix_voices_from_refs_inner`).
    pub fn process_chunk(&mut self, buffer: &mut [f32]) {
        // Skip if fully dry.
        if self.config.reverb_mix < 1e-6 {
            return;
        }
        if self.is_stereo {
            self.inner.process_stereo(buffer);
        } else {
            self.inner.process_mono(buffer);
        }
    }

    /// Reset the reverb state (clears all delay lines).
    ///
    /// Call this at the start of a new streaming session to prevent the
    /// reverb tail from a previous session bleeding into new audio.
    pub fn reset(&mut self) {
        self.inner = StereoReverb::new(&self.config);
    }

    /// The reverb configuration.
    #[must_use]
    pub fn config(&self) -> &ReverbConfig {
        &self.config
    }

    /// Whether this reverb processes stereo (interleaved) audio.
    #[must_use]
    pub fn is_stereo(&self) -> bool {
        self.is_stereo
    }

    /// Apply streaming reverb to a sequence of `AudioChunk`s in order.
    ///
    /// Processes each chunk's PCM in sequence, preserving reverb state
    /// across chunks. This is useful for the push-based streaming path
    /// where all chunks are produced at once: set `ChorusConfig.reverb`
    /// to `None` (to skip per-chunk batch reverb) and then call this on
    /// the assembled output.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError` if the reverb config is invalid.
    pub fn apply_to_chunks(&mut self, chunks: &mut [AudioChunk]) {
        for chunk in chunks.iter_mut() {
            self.process_chunk(&mut chunk.pcm);
        }
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_reverb_streaming_tests.rs"]
mod tests;
