// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pull-based streaming synthesis session for [`CompiledKokoro`].
//!
//! Unlike the push-based [`synthesize_streaming`](super::CompiledKokoro::synthesize_streaming)
//! which synthesizes all chunks at once and returns a `Vec<AudioChunk>`,
//! `StreamingKokoroSession` is a state machine that synthesizes one chunk per
//! `next_chunk()` call. The caller controls pacing, which is ideal for chorus
//! synthesis where multiple voices must be coordinated.
//!
//! # Design
//!
//! `CompiledKokoro` is `!Send` (it contains `RefCell<Option<MetalBuffer>>`),
//! so we cannot move it to a background thread. Instead, the session owns
//! only the pre-tokenized chunks and synthesis parameters. The caller passes
//! `&mut CompiledKokoro` and `&PipelineCache` to each `next_chunk()` call,
//! keeping ownership and lifetime management entirely in the caller's hands.
//!
//! # Usage
//!
//! ```text
//! let chunks: Vec<(DynTensor, DynTensor)> = prepare_chunks(...);
//! let mut session = StreamingKokoroSession::new(chunks, 1.0);
//!
//! while let Some(result) = session.next_chunk(&mut kokoro, &cache) {
//!     let (audio, cert) = result?;
//!     // Process audio chunk immediately (play, mix, etc.)
//! }
//! ```
//!
//! Part of #4105.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_models::kokoro_streaming::{AudioChunk, KokoroStreamConfig, StreamingAssembler};
use nn_tts_verify::Certificate;

use super::precompile::PrecompileShapes;
use super::CompiledKokoro;
use super::CompiledKokoroError;
use crate::cache::PipelineCache;

/// Pull-based streaming synthesis session.
///
/// Holds pre-tokenized chunks and synthesizes them one at a time on demand.
/// The caller provides `&mut CompiledKokoro` and `&PipelineCache` to each
/// `next_chunk()` call, sidestepping the `!Send` constraint on `CompiledKokoro`.
///
/// # State machine
///
/// The session tracks a cursor into a `Vec<(DynTensor, DynTensor)>` of
/// `(input_ids, style)` pairs. Each `next_chunk()` call advances the cursor
/// by one and synthesizes the chunk at the previous cursor position. When the
/// cursor reaches the end, `next_chunk()` returns `None`.
pub struct StreamingKokoroSession {
    /// Pre-tokenized chunks: `(input_ids, style)` pairs.
    chunks: Vec<(DynTensor, DynTensor)>,
    /// Speaking rate multiplier, passed to `CompiledKokoro::synthesize`.
    speed: f32,
    /// Current position in the chunk sequence. Starts at 0, increments on
    /// each `next_chunk()` call. When `cursor >= chunks.len()`, the session
    /// is done.
    cursor: usize,
    /// Whether to precompile pipeline segments for the chunk shapes before
    /// the first synthesis. Defaults to `true`. Set to `false` via
    /// [`with_precompile`](Self::with_precompile) to skip warmup for
    /// latency-sensitive cold starts where compilation-on-demand is
    /// acceptable.
    precompile: bool,
    /// Tracks whether precompilation has already been performed so it only
    /// runs once (on the first [`next_chunk`](Self::next_chunk) call or
    /// via an explicit [`run_precompile`](Self::run_precompile) invocation).
    precompiled: bool,
    /// Optional crossfade assembler for [`next_chunk_crossfaded`]. When
    /// present, raw PCM from each synthesis is fed through the assembler
    /// to apply crossfade (Hann or linear, per config) at chunk boundaries.
    /// Created by [`from_token_ids`] or [`with_crossfade`].
    assembler: Option<StreamingAssembler>,
}

impl StreamingKokoroSession {
    /// Create a new streaming session from pre-chunked token sequences.
    ///
    /// Each chunk is a `(input_ids, style)` pair where `input_ids` is a
    /// `DynTensor` of shape `[1, T_i]` and `style` is a `DynTensor` of
    /// shape `[1, 2*style_dim]`.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-tokenized `(input_ids, style)` pairs.
    /// * `speed` - Speaking rate multiplier (shared across all chunks).
    #[must_use]
    pub fn new(chunks: Vec<(DynTensor, DynTensor)>, speed: f32) -> Self {
        Self {
            chunks,
            speed,
            cursor: 0,
            precompile: true,
            precompiled: false,
            assembler: None,
        }
    }

    /// Create a streaming session from raw token ID vectors.
    ///
    /// Converts each `Vec<i64>` into a `DynTensor` of shape `[1, T_i]`
    /// with dtype `I64`, paired with the shared `style` tensor. A
    /// [`StreamingAssembler`] is created for crossfade at chunk boundaries,
    /// usable via [`next_chunk_crossfaded`](Self::next_chunk_crossfaded).
    ///
    /// # Arguments
    ///
    /// * `chunks` - Raw token ID sequences per chunk.
    /// * `style` - Style embedding `[1, 2*style_dim]` shared across chunks.
    /// * `speed` - Speaking rate multiplier.
    /// * `stream_config` - Crossfade configuration for chunk boundaries.
    pub fn from_token_ids(
        chunks: Vec<Vec<i64>>,
        style: DynTensor,
        speed: f32,
        stream_config: KokoroStreamConfig,
    ) -> Result<Self, CompiledKokoroError> {
        if chunks.is_empty() {
            return Ok(Self {
                chunks: Vec::new(),
                speed,
                cursor: 0,
                precompile: true,
                precompiled: false,
                assembler: None,
            });
        }

        let cpu = Device::Cpu;
        let mut tensor_chunks = Vec::with_capacity(chunks.len());
        for token_ids in &chunks {
            let len = token_ids.len();
            let input_ids = DynTensor::from_vec_i64(token_ids.clone(), &[1, len], &cpu)
                .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
            tensor_chunks.push((input_ids, style.clone()));
        }

        let total = tensor_chunks.len();
        let assembler = StreamingAssembler::new(stream_config, total)
            .map_err(|e| CompiledKokoroError::InvalidInput(e.to_string()))?;

        Ok(Self {
            chunks: tensor_chunks,
            speed,
            cursor: 0,
            precompile: true,
            precompiled: false,
            assembler: Some(assembler),
        })
    }

    /// Attach a crossfade assembler to an existing session.
    ///
    /// After calling this, [`next_chunk_crossfaded`](Self::next_chunk_crossfaded)
    /// will apply crossfade at chunk boundaries. The session must not have
    /// started synthesis yet (cursor == 0).
    pub fn with_crossfade(
        mut self,
        stream_config: KokoroStreamConfig,
    ) -> Result<Self, CompiledKokoroError> {
        let total = self.chunks.len();
        if total == 0 {
            return Ok(self);
        }
        let assembler = StreamingAssembler::new(stream_config, total)
            .map_err(|e| CompiledKokoroError::InvalidInput(e.to_string()))?;
        self.assembler = Some(assembler);
        Ok(self)
    }

    /// Whether a crossfade assembler is attached.
    #[must_use]
    pub fn has_crossfade(&self) -> bool {
        self.assembler.is_some()
    }

    /// Set whether to precompile pipeline segments for the chunk shapes
    /// before the first synthesis.
    ///
    /// When `true` (the default), the first [`next_chunk`](Self::next_chunk)
    /// call runs [`CompiledKokoro::warmup`] with shapes tailored to the
    /// actual token lengths of the chunks in this session. This eliminates
    /// per-chunk compilation latency at the cost of a one-time warmup pass.
    ///
    /// Set to `false` for latency-sensitive cold starts where the caller
    /// prefers immediate first-chunk synthesis with on-demand compilation.
    #[must_use]
    pub fn with_precompile(mut self, precompile: bool) -> Self {
        self.precompile = precompile;
        self
    }

    /// Whether shape precompilation is enabled for this session.
    #[must_use]
    pub fn precompile_enabled(&self) -> bool {
        self.precompile
    }

    /// Explicitly run shape precompilation.
    ///
    /// Extracts the token lengths from all chunks, generates a tailored
    /// [`PrecompileShapes`], and calls [`CompiledKokoro::warmup`] to
    /// pre-compile the pipeline segments for those shapes.
    ///
    /// This is called automatically on the first [`next_chunk`](Self::next_chunk)
    /// when precompilation is enabled. Use this method to control when the
    /// warmup happens (e.g., during an explicit loading phase).
    ///
    /// Returns the number of segment compilations performed, or 0 if
    /// there are no chunks or precompilation was already performed.
    pub fn run_precompile(
        &mut self,
        kokoro: &mut CompiledKokoro,
        cache: &PipelineCache,
    ) -> Result<usize, CompiledKokoroError> {
        if self.precompiled {
            return Ok(0);
        }
        self.precompiled = true;

        let token_lengths: Vec<usize> = self
            .chunks
            .iter()
            .filter_map(|(input_ids, _)| {
                let dims = input_ids.dims();
                // input_ids is [1, T] — extract T.
                if dims.len() >= 2 { Some(dims[1]) } else { None }
            })
            .collect();

        let shapes = match PrecompileShapes::from_token_lengths(&token_lengths) {
            Some(s) => s,
            None => return Ok(0),
        };

        kokoro.warmup(&shapes, cache)
    }

    /// Synthesize the next chunk. Returns `None` when all chunks are consumed.
    ///
    /// The caller provides `&mut CompiledKokoro` and `&PipelineCache` for
    /// each call. This design avoids holding a reference to `CompiledKokoro`
    /// in the session, sidestepping the `!Send` constraint.
    ///
    /// On the first call, if precompilation is enabled (the default), this
    /// runs [`CompiledKokoro::warmup`] with shapes tailored to the chunks.
    ///
    /// # Returns
    ///
    /// - `Some(Ok((audio, certificate)))` — the synthesized audio and its
    ///   verification certificate for the current chunk.
    /// - `Some(Err(..))` — synthesis failed for the current chunk. The cursor
    ///   is still advanced; call `next_chunk()` again to attempt the next chunk.
    /// - `None` — all chunks have been consumed.
    pub fn next_chunk(
        &mut self,
        kokoro: &mut CompiledKokoro,
        cache: &PipelineCache,
    ) -> Option<Result<(DynTensor, Certificate), CompiledKokoroError>> {
        if self.cursor >= self.chunks.len() {
            return None;
        }

        // Reset SineGen phase at the start of a new streaming session so the
        // first chunk begins with zero phase. Subsequent chunks carry phase
        // continuity automatically via build_harmonic_source.
        if self.cursor == 0 {
            kokoro.reset_sinegen_phase();
        }

        // Run precompilation once before the first synthesis.
        if self.precompile && !self.precompiled {
            if let Err(e) = self.run_precompile(kokoro, cache) {
                // Precompilation failure is non-fatal — synthesis will
                // compile on demand for any missed shapes.
                let _ = e;
            }
        }

        let idx = self.cursor;
        self.cursor += 1;
        let (ref input_ids, ref style) = self.chunks[idx];
        Some(kokoro.synthesize(input_ids, style, self.speed, cache))
    }

    /// Synthesize the next chunk with crossfade applied.
    ///
    /// Like [`next_chunk`](Self::next_chunk), but feeds the raw PCM through
    /// the internal [`StreamingAssembler`] to apply crossfade (Hann or
    /// linear, per config) at chunk boundaries. Returns an [`AudioChunk`] with crossfaded PCM
    /// ready for immediate playback.
    ///
    /// Requires a crossfade assembler (created by [`from_token_ids`] or
    /// [`with_crossfade`]). Returns `Some(Err(..))` if no assembler is
    /// attached.
    ///
    /// # Returns
    ///
    /// - `Some(Ok(AudioChunk))` — crossfaded audio chunk.
    /// - `Some(Err(..))` — synthesis or assembly failed.
    /// - `None` — all chunks have been consumed.
    pub fn next_chunk_crossfaded(
        &mut self,
        kokoro: &mut CompiledKokoro,
        cache: &PipelineCache,
    ) -> Option<Result<AudioChunk, CompiledKokoroError>> {
        let result = self.next_chunk(kokoro, cache)?;
        let (audio_tensor, _cert) = match result {
            Ok(pair) => pair,
            Err(e) => return Some(Err(e)),
        };

        let assembler = match self.assembler.as_mut() {
            Some(a) => a,
            None => {
                return Some(Err(CompiledKokoroError::InvalidInput(
                    "next_chunk_crossfaded requires a crossfade assembler \
                     (use from_token_ids or with_crossfade)"
                        .into(),
                )));
            }
        };

        let pcm = match audio_tensor.to_flat_vec::<f32>() {
            Ok(v) => v,
            Err(e) => return Some(Err(CompiledKokoroError::Tensor(Box::new(e)))),
        };

        match assembler.push_raw(pcm) {
            Ok(chunk) => Some(Ok(chunk)),
            Err(e) => Some(Err(CompiledKokoroError::InvalidInput(e.to_string()))),
        }
    }

    /// Number of remaining chunks (not yet synthesized).
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.chunks.len().saturating_sub(self.cursor)
    }

    /// Whether all chunks have been synthesized.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.cursor >= self.chunks.len()
    }

    /// Total number of chunks in this session.
    #[must_use]
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Number of chunks already synthesized.
    #[must_use]
    pub fn synthesized_count(&self) -> usize {
        self.cursor
    }

    /// Reset the session to re-synthesize from the beginning.
    ///
    /// Useful for re-synthesizing the same text with a different voice
    /// (different `CompiledKokoro` instance) without re-tokenizing.
    /// Precompilation state is preserved — if shapes were already warmed
    /// up, they remain cached.
    pub fn reset(&mut self) {
        self.cursor = 0;
        if let Some(ref mut asm) = self.assembler {
            asm.reset();
        }
    }

    /// Update the speaking rate for subsequent chunks.
    ///
    /// Already-synthesized chunks are not affected.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Current speaking rate.
    #[must_use]
    pub fn speed(&self) -> f32 {
        self.speed
    }
}

#[cfg(kani)]
impl StreamingKokoroSession {
    /// Create a session with a controlled chunk count and cursor position
    /// for Kani verification. Uses CPU-only dummy DynTensors that are never
    /// actually synthesized — only the state machine fields matter.
    pub(crate) fn kani_from_len_cursor(len: usize, cursor: usize) -> Self {
        let cpu = nn_core::Device::Cpu;
        let chunks: Vec<(DynTensor, DynTensor)> = (0..len)
            .map(|_| {
                let ids = DynTensor::zeros(&[1, 1], nn_core::DType::I64, &cpu).unwrap();
                let sty = DynTensor::zeros(&[1, 1], nn_core::DType::F32, &cpu).unwrap();
                (ids, sty)
            })
            .collect();
        Self {
            chunks,
            speed: 1.0,
            cursor,
            precompile: false,
            precompiled: false,
            assembler: None,
        }
    }

    /// Advance the cursor by 1 (models what `next_chunk` does to the cursor).
    /// Does NOT call synthesize — only advances cursor state.
    pub(crate) fn kani_advance_cursor(&mut self) {
        if self.cursor < self.chunks.len() {
            self.cursor += 1;
        }
    }
}

#[cfg(test)]
#[path = "compiled_kokoro_pull_streaming_tests.rs"]
mod tests;
