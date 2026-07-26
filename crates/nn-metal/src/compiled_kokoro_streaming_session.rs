// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU streaming session for chunk-by-chunk Kokoro synthesis.
//!
//! [`CompiledKokoroStreamingSession`] provides a pull-based iterator-style API
//! over [`CompiledKokoro`], synthesizing one audio chunk per `next_chunk()` call.
//! Each chunk is ready for immediate playback as soon as it returns — the caller
//! does not wait for the full utterance to finish before hearing audio.
//!
//! # Architecture
//!
//! ```text
//! CompiledKokoroStreamingSession::new(chunks, style, speed, stream_config)
//!   → session { chunks, cursor, assembler }
//!
//! loop {
//!     session.next_chunk(&mut kokoro, &cache)?  → Some(AudioChunk)
//!     play(chunk.pcm);   // immediate playback
//! }
//! // → None when all chunks synthesized
//! ```
//!
//! Internally, each `next_chunk()` call runs the full 8-segment Kokoro pipeline
//! (steps 1-8) for one token chunk, then feeds the raw PCM through a
//! [`StreamingAssembler`] for crossfade at chunk boundaries.
//!
//! # Callback-driven streaming
//!
//! For push-based usage, use [`CompiledKokoro::synthesize_streaming_callback`]
//! which takes an `on_chunk` closure called after each chunk is synthesized.
//!
//! Part of #4105, #3351.

use nn_core::dyn_tensor::DynTensor;
use nn_models::kokoro_error::validate_speed;
use nn_models::kokoro_streaming::{AudioChunk, KokoroStreamConfig, StreamingAssembler};

use super::{validate_input_ids, CompiledKokoro, CompiledKokoroError};
use crate::cache::PipelineCache;

/// Pull-based GPU streaming session for chunk-by-chunk Kokoro synthesis.
///
/// Pre-accepts chunked token IDs and synthesis parameters at construction.
/// Each `next_chunk()` call synthesizes one chunk via `CompiledKokoro` and
/// returns a crossfaded [`AudioChunk`] ready for playback.
///
/// # Example
///
/// ```rust,ignore
/// let session = CompiledKokoroStreamingSession::new(
///     chunks, &style, 1.0, KokoroStreamConfig::default(),
/// )?;
/// while let Some(chunk) = session.next_chunk(&mut kokoro, &cache)? {
///     audio_queue.enqueue(&chunk.pcm);
/// }
/// ```
///
/// # Memory
///
/// Only retains the crossfade tail of the previous chunk (up to
/// `crossfade_samples` floats). The session itself does not accumulate
/// audio — each chunk is independently playable.
pub struct CompiledKokoroStreamingSession {
    /// Pre-tokenized input chunks (token IDs per chunk).
    chunks: Vec<DynTensor>,
    /// Style embedding `[1, 2*style_dim]` shared across all chunks.
    style: DynTensor,
    /// Speaking rate multiplier shared across all chunks.
    speed: f32,
    /// Current chunk index (next to synthesize).
    cursor: usize,
    /// Crossfade assembler state.
    assembler: StreamingAssembler,
}

impl CompiledKokoroStreamingSession {
    /// Create a new streaming session from pre-tokenized chunks.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-tokenized token ID tensors, each `[1, T_i]`.
    /// * `style` - Style embedding `[1, 2*style_dim]` shared across all chunks.
    /// * `speed` - Speaking rate multiplier (1.0 = normal).
    /// * `stream_config` - Crossfade configuration for chunk boundaries.
    ///
    /// # Errors
    ///
    /// Returns error if `chunks` is empty, `speed` is invalid, or
    /// `stream_config` validation fails.
    pub fn new(
        chunks: Vec<DynTensor>,
        style: DynTensor,
        speed: f32,
        stream_config: KokoroStreamConfig,
    ) -> Result<Self, CompiledKokoroError> {
        if chunks.is_empty() {
            return Err(CompiledKokoroError::InvalidInput(
                "streaming session requires at least one chunk".into(),
            ));
        }
        validate_speed(speed)?;
        let total = chunks.len();
        let assembler = StreamingAssembler::new(stream_config, total)?;
        Ok(Self {
            chunks,
            style,
            speed,
            cursor: 0,
            assembler,
        })
    }

    /// Synthesize and return the next audio chunk, or `None` if complete.
    ///
    /// Each call runs the full Kokoro GPU pipeline (steps 1-8) for one
    /// token chunk and applies crossfade with the previous chunk's tail.
    /// The returned [`AudioChunk`] is ready for immediate playback.
    ///
    /// # Arguments
    ///
    /// * `kokoro` - Compiled Kokoro pipeline (mutable for segment cache updates).
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(AudioChunk))` — next synthesized + crossfaded chunk.
    /// - `Ok(None)` — all chunks have been synthesized.
    /// - `Err(CompiledKokoroError)` — synthesis or assembly failed.
    pub fn next_chunk(
        &mut self,
        kokoro: &mut CompiledKokoro,
        cache: &PipelineCache,
    ) -> Result<Option<AudioChunk>, CompiledKokoroError> {
        if self.cursor >= self.chunks.len() {
            return Ok(None);
        }

        // Reset SineGen phase at the start of a new streaming session so the
        // first chunk begins with zero phase. Subsequent chunks carry phase
        // continuity automatically via build_harmonic_source.
        if self.cursor == 0 {
            kokoro.reset_sinegen_phase();
        }

        let chunk_input = &self.chunks[self.cursor];
        let (audio_tensor, _cert) =
            kokoro.synthesize(chunk_input, &self.style, self.speed, cache)?;
        let pcm = audio_tensor
            .to_flat_vec::<f32>()
            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;

        let audio_chunk = self
            .assembler
            .push_raw(pcm)
            .map_err(CompiledKokoroError::from)?;

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
    /// After calling `cancel()`, `remaining()` returns 0 and `next_chunk()`
    /// returns `Ok(None)`. Useful for aborting synthesis mid-stream (e.g.,
    /// user interrupts playback).
    pub fn cancel(&mut self) {
        self.cursor = self.chunks.len();
    }
}

impl CompiledKokoro {
    /// Synthesize chunked token sequences with per-chunk callback.
    ///
    /// Like [`synthesize_streaming`](Self::synthesize_streaming) but calls
    /// `on_chunk` after each chunk is synthesized and crossfaded, enabling
    /// playback before the full utterance finishes. This is the key API for
    /// real-time TTS: the first audio chunk is delivered after one synthesis
    /// pass, not after all chunks complete.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-chunked token IDs as `DynTensor`s, each `[1, T_i]`.
    /// * `style` - Style embedding `[1, 2*style_dim]` (shared across all chunks).
    /// * `speed` - Speaking rate multiplier (shared across all chunks).
    /// * `stream_config` - Crossfade configuration.
    /// * `cache` - Metal pipeline cache.
    /// * `on_chunk` - Called with each [`AudioChunk`] as soon as it is ready.
    ///   The callback receives chunks in order (chunk 0 first, then 1, etc.).
    ///   Return `false` from the callback to abort synthesis early.
    ///
    /// # Returns
    ///
    /// The total number of chunks synthesized (may be less than `chunks.len()`
    /// if the callback returned `false` to abort early).
    ///
    /// # Errors
    ///
    /// Returns error if any chunk fails to synthesize, or if chunks are too
    /// short for the configured crossfade overlap.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut total_samples = 0;
    /// let n = kokoro.synthesize_streaming_callback(
    ///     &chunks, &style, 1.0, &stream_config, &cache,
    ///     |chunk| {
    ///         audio_queue.enqueue(&chunk.pcm);
    ///         total_samples += chunk.pcm.len();
    ///         true // continue
    ///     },
    /// )?;
    /// ```
    pub fn synthesize_streaming_callback(
        &mut self,
        chunks: &[DynTensor],
        style: &DynTensor,
        speed: f32,
        stream_config: &KokoroStreamConfig,
        cache: &PipelineCache,
        mut on_chunk: impl FnMut(&AudioChunk) -> bool,
    ) -> Result<usize, CompiledKokoroError> {
        if chunks.is_empty() {
            return Ok(0);
        }
        validate_speed(speed)?;

        // Reset SineGen phase for this new streaming session.
        self.reset_sinegen_phase();

        let total = chunks.len();
        let mut assembler = StreamingAssembler::new(stream_config.clone(), total)?;
        let mut synthesized = 0;

        for chunk_input in chunks {
            let (audio_tensor, _cert) = self.synthesize(chunk_input, style, speed, cache)?;
            let pcm = audio_tensor
                .to_flat_vec::<f32>()
                .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;

            let audio_chunk = assembler.push_raw(pcm)?;
            synthesized += 1;

            if !on_chunk(&audio_chunk) {
                // Caller requested early abort.
                break;
            }
        }

        Ok(synthesized)
    }

    /// Create a pull-based streaming session for chunk-by-chunk synthesis.
    ///
    /// Returns a [`CompiledKokoroStreamingSession`] that yields one
    /// [`AudioChunk`] per `next_chunk()` call. The session holds the token
    /// chunks, style, and speed internally — the caller only needs to provide
    /// `&mut self` and `&PipelineCache` on each iteration.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-tokenized token ID tensors, each `[1, T_i]`.
    /// * `style` - Style embedding `[1, 2*style_dim]` shared across all chunks.
    /// * `speed` - Speaking rate multiplier (1.0 = normal).
    /// * `stream_config` - Crossfade configuration for chunk boundaries.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut session = kokoro.create_streaming_session(
    ///     chunks, style.clone(), 1.0, KokoroStreamConfig::default(),
    /// )?;
    /// while let Some(chunk) = session.next_chunk(&mut kokoro, &cache)? {
    ///     audio_queue.enqueue(&chunk.pcm);
    /// }
    /// ```
    pub fn create_streaming_session(
        &self,
        chunks: Vec<DynTensor>,
        style: DynTensor,
        speed: f32,
        stream_config: KokoroStreamConfig,
    ) -> Result<CompiledKokoroStreamingSession, CompiledKokoroError> {
        // Validate input_ids against model config before creating session.
        let max_pos = self.config().plbert.max_position_embeddings;
        for (i, chunk) in chunks.iter().enumerate() {
            validate_input_ids(chunk, max_pos).map_err(|e| match e {
                CompiledKokoroError::InvalidInput(msg) => {
                    CompiledKokoroError::InvalidInput(format!("chunk[{i}]: {msg}"))
                }
                other => other,
            })?;
        }
        CompiledKokoroStreamingSession::new(chunks, style, speed, stream_config)
    }
}
