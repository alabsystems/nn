// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pull-based streaming synthesis session for Kokoro TTS.
//!
//! Provides [`StreamingKokoroSession`], an iterator-like struct that
//! pre-tokenizes text into chunks at construction, then yields one
//! [`AudioChunk`] per [`next_chunk()`](StreamingKokoroSession::next_chunk)
//! call. Crossfade between adjacent chunks is applied automatically via
//! [`StreamingAssembler`].
//!
//! Unlike the push-based [`text_to_audio_streaming()`] which takes a callback
//! and drives synthesis internally, this API lets the caller control iteration
//! -- essential for dvoice's conductor, which interleaves synthesis with
//! playback scheduling and chorus coordination.
//!
//! # Architecture
//!
//! ```text
//! KokoroTextPipeline::create_streaming_session(text, phonemize, config)
//!     → StreamingKokoroSession { chunks, cursor, assembler }
//!
//! loop {
//!     session.next_chunk(&mut synth, &style, speed)?  → Some(AudioChunk)
//!     play(chunk.pcm());
//! }
//! // → None when complete
//! ```
//!
//! Part of #3351 (Phase 3C).

use super::{AudioChunk, KokoroStreamConfig, StreamingAssembler};
use crate::kokoro_error::KokoroError;
use crate::kokoro_pipeline::KokoroSynth;
use nn_core::dyn_tensor::DynTensor;

/// Pull-based streaming synthesis session.
///
/// Pre-tokenizes text into chunks at construction, then synthesizes one
/// chunk per `next_chunk()` call. Crossfade is applied automatically
/// between adjacent chunks via [`StreamingAssembler`].
///
/// # Example
///
/// ```rust,ignore
/// let session = StreamingKokoroSession::new(chunks, stream_config)?;
/// while let Some(chunk) = session.next_chunk(&mut synth, &style, speed)? {
///     play(chunk.pcm());
/// }
/// ```
///
/// # Error Handling
///
/// The session converts backend-specific synthesis errors into
/// [`KokoroError`] via the `Into<KokoroError>` bound on the synth's error
/// type. This allows the caller to handle all errors uniformly.
#[derive(Debug)]
pub struct StreamingKokoroSession {
    /// Pre-tokenized input chunks (token IDs per chunk).
    chunks: Vec<DynTensor>,
    /// Current chunk index (next to synthesize).
    cursor: usize,
    /// Crossfade assembler state.
    assembler: StreamingAssembler,
}

impl StreamingKokoroSession {
    /// Create a new session from pre-tokenized chunks.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-tokenized token ID tensors, each `[1, T_i]`.
    ///   Use `KokoroTextPipeline::text_to_tokens()` + `chunks_to_tensors()`
    ///   to produce these from raw text.
    /// * `stream_config` - Crossfade configuration for chunk boundaries.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if `chunks` is empty or if
    /// `stream_config` validation fails.
    pub fn new(
        chunks: Vec<DynTensor>,
        stream_config: KokoroStreamConfig,
    ) -> Result<Self, KokoroError> {
        if chunks.is_empty() {
            return Err(KokoroError::InvalidConfig {
                field: "chunks",
                reason: "must contain at least one chunk".into(),
            });
        }
        let total = chunks.len();
        let assembler = StreamingAssembler::new(stream_config, total)?;
        Ok(Self {
            chunks,
            cursor: 0,
            assembler,
        })
    }

    /// Synthesize and return the next audio chunk, or `None` if complete.
    ///
    /// The `synth` backend performs actual synthesis (CPU or GPU). The
    /// session feeds the raw PCM through the internal [`StreamingAssembler`]
    /// to apply crossfade at chunk boundaries.
    ///
    /// # Type Parameters
    ///
    /// The synth's `Error` type must convert into `KokoroError` so that
    /// all errors surface uniformly. Both `KokoroModel` (CPU) and
    /// `CompiledKokoro` (GPU) satisfy this via their `KokoroError` error type.
    ///
    /// # Arguments
    ///
    /// * `synth` - Synthesis backend (implements [`KokoroSynth`]).
    /// * `style` - Style embedding `[1, 2*style_dim]` shared across chunks.
    /// * `speed` - Speaking rate multiplier shared across chunks.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(AudioChunk))` — next synthesized + crossfaded chunk.
    /// - `Ok(None)` — all chunks have been synthesized.
    /// - `Err(KokoroError)` — synthesis or assembly failed.
    pub fn next_chunk<S: KokoroSynth>(
        &mut self,
        synth: &mut S,
        style: &DynTensor,
        speed: f32,
    ) -> Result<Option<AudioChunk>, KokoroError>
    where
        S::Error: Into<KokoroError>,
    {
        if self.cursor >= self.chunks.len() {
            return Ok(None);
        }

        // Reset SineGen phase at the start of a new streaming session so the
        // first chunk begins with zero phase. Subsequent chunks carry phase
        // continuity automatically via SineGen's stateful cumulative sum.
        if self.cursor == 0 {
            synth.reset_sinegen_phase();
        }

        // Synthesize raw PCM for the current chunk.
        let raw_pcm = synth
            .synthesize_chunk(&self.chunks[self.cursor], style, speed)
            .map_err(Into::into)?;

        // Feed through assembler for crossfade.
        let audio_chunk = self.assembler.push_raw(raw_pcm)?;

        self.cursor += 1;

        Ok(Some(audio_chunk))
    }

    /// Number of chunks remaining (including the current one).
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.chunks.len().saturating_sub(self.cursor)
    }

    /// Total number of chunks in the session.
    #[must_use]
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Whether all chunks have been synthesized.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.cursor >= self.chunks.len()
    }

    /// Current cursor position (0-based index of the next chunk to synthesize).
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Cancel the session, consuming all remaining chunks.
    ///
    /// After calling `cancel()`, `remaining()` returns 0, `is_complete()`
    /// returns `true`, and `next_chunk()` returns `Ok(None)`.
    ///
    /// This is useful for dvoice's conductor when the user interrupts
    /// playback mid-stream — the session can be cleanly discarded without
    /// synthesizing remaining chunks.
    pub fn cancel(&mut self) {
        self.cursor = self.chunks.len();
    }
}
