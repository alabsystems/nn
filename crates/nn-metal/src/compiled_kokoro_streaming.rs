// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming synthesis for [`CompiledKokoro`].
//!
//! Synthesizes chunked token sequences with crossfade at chunk boundaries,
//! producing a sequence of [`AudioChunk`]s suitable for incremental playback.
//!
//! # Usage
//!
//! ```text
//! tokenizer.chunk_and_encode(phonemes)
//!   → Vec<(String, Vec<u32>)>          // phoneme chunks + token IDs
//!   → CompiledKokoro::synthesize_streaming()
//!   → Vec<AudioChunk>                  // crossfaded, playable chunks
//! ```
//!
//! Each chunk is synthesized independently via `synthesize()`, then
//! `assemble_streaming_chunks()` applies crossfade at boundaries (Hann
//! window for >= 40ms overlap, linear for shorter overlaps).
//!
//! Part of #3355, #3351, #2918.

use nn_core::dyn_tensor::DynTensor;
use nn_models::kokoro_streaming::{assemble_streaming_chunks, AudioChunk, KokoroStreamConfig};

use super::*;

impl CompiledKokoro {
    /// Synthesize chunked token sequences with crossfade.
    ///
    /// Each `(input_ids, style)` pair is synthesized independently, then
    /// adjacent chunks are crossfaded using the streaming config's overlap.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-chunked token IDs as `DynTensor`s, each `[1, T_i]`.
    ///   Use `KokoroTokenizer::chunk_and_encode()` to split long text.
    /// * `style` - Style embedding `[1, 2*style_dim]` (shared across all chunks).
    /// * `speed` - Speaking rate multiplier (shared across all chunks).
    /// * `stream_config` - Crossfade configuration.
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// A `Vec<AudioChunk>` with crossfade applied at all boundaries.
    /// Each chunk's `pcm` is ready for immediate playback.
    ///
    /// # Errors
    ///
    /// Returns error if any chunk fails to synthesize, or if chunks are too
    /// short for the configured crossfade overlap.
    pub fn synthesize_streaming(
        &mut self,
        chunks: &[DynTensor],
        style: &DynTensor,
        speed: f32,
        stream_config: &KokoroStreamConfig,
        cache: &PipelineCache,
    ) -> Result<Vec<AudioChunk>, CompiledKokoroError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        // Reset SineGen phase for this new streaming session. Within the loop,
        // build_harmonic_source carries the terminal cumulative phase from each
        // chunk to the next, ensuring phase continuity at chunk boundaries.
        self.reset_sinegen_phase();

        // Synthesize each chunk with SineGen phase continuity. GPU work from
        // consecutive chunks is pipelined by Metal's lazy command buffer batching.
        let mut raw_chunks: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        for chunk_input in chunks {
            let (audio_tensor, _cert) = self.synthesize(chunk_input, style, speed, cache)?;
            let pcm = audio_tensor
                .to_flat_vec::<f32>()
                .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
            raw_chunks.push(pcm);
        }

        // Assemble with crossfade.
        Ok(assemble_streaming_chunks(&raw_chunks, stream_config)?)
    }
}
