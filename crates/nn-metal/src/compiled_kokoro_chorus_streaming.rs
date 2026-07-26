// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming chorus synthesis methods for [`KokoroChorus`].
//!
//! Extracted from `compiled_kokoro_chorus.rs` to comply with the 500-line
//! limit. These methods implement chunk-by-chunk synthesis with per-chunk
//! voice mixing and crossfade at chunk boundaries.
//!
//! Part of #3403, #3351 (T4.2).

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_models::kokoro_chorus_reverb::ReverbConfig;
use nn_models::kokoro_chorus_reverb_streaming::StreamingReverb;
use nn_models::kokoro_error::{validate_speed, KokoroError};
use nn_models::kokoro_streaming::{AudioChunk, KokoroStreamConfig, StreamingAssembler};

use super::chorus::{
    contextualize_invalid_input, run_voice_pipeline, validate_indexed_input_ids,
    verify_and_extract_pcm, DefaultArenaCheckpoint, KokoroChorus,
};
use super::{gpu, validate_input_ids, CompiledKokoroError};
use crate::cache::PipelineCache;

impl KokoroChorus {
    /// Synthesize a streaming chorus: chunked synthesis per voice, per-chunk
    /// mixing, and crossfade at chunk boundaries.
    ///
    /// All voices share the same text chunks (different styles). Delegates to
    /// [`synthesize_streaming_chorus_shared_encode`] which encodes each chunk
    /// once and runs Steps 3-8 per-voice with the shared encoding.
    ///
    /// First-audio latency = 1 chunk × N voices (not full utterance × N voices).
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-chunked token IDs as `DynTensor`s, each `[1, T_i]`.
    ///   All voices synthesize the same chunks (same text, different styles).
    /// * `styles` - Per-voice style embeddings. Length must equal `n_voices`.
    /// * `speed` - Speaking rate multiplier (shared).
    /// * `stream_config` - Crossfade configuration.
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// `Vec<AudioChunk>` with per-chunk voice mixing and crossfade applied.
    pub fn synthesize_streaming_chorus(
        &mut self,
        chunks: &[DynTensor],
        styles: &[DynTensor],
        speed: f32,
        stream_config: &KokoroStreamConfig,
        cache: &PipelineCache,
    ) -> Result<Vec<AudioChunk>, CompiledKokoroError> {
        self.synthesize_streaming_chorus_shared_encode(chunks, styles, speed, stream_config, cache)
    }

    /// Streaming chorus with per-chunk shared encoding (same text, different styles).
    ///
    /// For each chunk: Steps 1-2 (PLBert + TextEncoder) execute once on
    /// `voices[0]`, then Steps 3-8 execute per-voice using the shared encoding.
    /// For 8 voices with K chunks, this eliminates K×7 redundant encoding
    /// passes (~66ms GPU saved per chunk at D=512, per #3229 profiling).
    ///
    /// GPU data flow per chunk: `step_encode` produces standalone GPU buffers
    /// (via `without_arena`) that survive across per-voice Steps 3-8. Each
    /// voice's work is encoded into the lazy batch; flushes occur only at
    /// `step_regulate` (4-byte scalar readback) and `step_istft` (terminal
    /// PCM readback).
    ///
    /// Part of #3351 (T4.2), design: `designs/2026-03-23-multi-voice-gpu-scheduling.md` Level 2.
    pub fn synthesize_streaming_chorus_shared_encode(
        &mut self,
        chunks: &[DynTensor],
        styles: &[DynTensor],
        speed: f32,
        stream_config: &KokoroStreamConfig,
        cache: &PipelineCache,
    ) -> Result<Vec<AudioChunk>, CompiledKokoroError> {
        let n = self.voices.len();
        if styles.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            ))
            .into());
        }
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        validate_speed(speed)?;
        let max_position_embeddings = self.voices[0].config().plbert.max_position_embeddings;
        validate_indexed_input_ids(chunks, max_position_embeddings, "chunks")?;

        // Reset SineGen phase on all voices at the start of a new streaming
        // session. Within the loop, build_harmonic_source carries the terminal
        // cumulative phase from each chunk to the next, ensuring phase
        // continuity at chunk boundaries.
        for voice in &mut self.voices {
            voice.reset_sinegen_phase();
        }

        // Reclaim buffer pool entries from previous synthesis call.
        crate::arena::pool_reclaim();

        // Pre-size the arena for the N-voice × K-chunk synthesis loop (#4289).
        // Multiply by 2 to account for encode + one voice's decode/regulate.
        // Per-voice checkpoint/restore reclaims between voices. Part of dvoice#2420.
        let arena_estimate = self.voices[0].estimate_arena_bytes().saturating_mul(2);
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(cache.context(), arena_estimate);
        }

        // Pre-split styles and upload to GPU (outside NaN skip scope).
        let mut decoder_styles: Vec<DynTensor> = Vec::with_capacity(n);
        let mut prosody_styles: Vec<DynTensor> = Vec::with_capacity(n);
        for style in styles {
            let split = self.voices[0].split_style(style)?;
            decoder_styles.push(split.decoder_style.to_device(&gpu())?);
            prosody_styles.push(split.prosody_style.to_device(&gpu())?);
        }

        let result = (|| -> Result<Vec<AudioChunk>, CompiledKokoroError> {
            // Determine channel count for crossfade assembler. Output is
            // stereo when the chorus pipeline, stereo config, or ChorusConfig
            // pans are present.
            let is_stereo = self.config.pans.is_some()
                || self.has_stereo()
                || self.has_chorus_pipeline()
                || self.has_spatial();
            let channels: usize = if is_stereo { 2 } else { 1 };

            // For stereo, double crossfade_samples so the crossfade covers
            // the same time duration (interleaved stereo has 2 floats per
            // time-domain sample).
            let effective_config = if is_stereo {
                KokoroStreamConfig::new(stream_config.crossfade_samples * 2)?
                    .with_window(stream_config.crossfade_window)
            } else {
                stream_config.clone()
            };
            let mut assembler =
                StreamingAssembler::new_with_channels(effective_config, chunks.len(), channels)?;
            let mut assembled_chunks: Vec<AudioChunk> = Vec::with_capacity(chunks.len());

            for chunk_input in chunks {
                validate_input_ids(chunk_input, max_position_embeddings)?;

                // Chunk-level NaN skip scope (matches synthesize() convention).
                let chunk_audios: Vec<DynTensor> = with_nan_check_policy(
                    NanCheckPolicy::Skip,
                    || -> Result<Vec<DynTensor>, CompiledKokoroError> {
                        // Steps 1-2: Encode this chunk once on voices[0].
                        let enc = self.voices[0].step_encode(chunk_input, cache)?;

                        // Per-voice: Steps 3-8 using shared encoding.
                        // Per-voice arena checkpoint/restore to prevent overflow
                        // thrashing across N voices. Part of dvoice#2420.
                        let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
                        for (i, voice) in self.voices.iter_mut().enumerate() {
                            let _arena_cp = DefaultArenaCheckpoint::new();
                            let audio = run_voice_pipeline(
                                voice,
                                &enc,
                                &prosody_styles[i],
                                &decoder_styles[i],
                                speed,
                                cache,
                            )?;
                            audios.push(audio);
                        }
                        Ok(audios)
                    },
                )?;

                // Outside Skip scope: NaN guard + verify per voice, extract PCM.
                let mut chunk_pcms = verify_and_extract_pcm(&self.voices, &chunk_audios)?;

                // Route through the full processing chain: detune, humanize,
                // per-voice EQ/de-essing, stereo/mono mix, bus EQ, dynamics
                // (or ChorusMasterPipeline when configured). Processing state
                // (compressor envelopes, EQ filters, limiter) persists on
                // `self` across chunks for smooth dynamics tracking.
                let mixed = self.mix_or_process(&mut chunk_pcms)?;

                // Optional crossfade boundary optimization: when a crossfade
                // optimizer is configured, push the mixed chunk through it to
                // find optimal crossfade points (zero-crossings, energy minima).
                let optimized = if let Some(ref mut optimizer) = self.crossfade_optimizer {
                    optimizer.push_chunk(&mixed).map_err(|e| {
                        CompiledKokoroError::InvalidInput(format!("crossfade optimizer: {e}"))
                    })?
                } else {
                    mixed
                };

                // Crossfade via StreamingAssembler (per-chunk incremental).
                let audio_chunk = assembler.push_raw(optimized)?;
                assembled_chunks.push(audio_chunk);
            }

            // Flush any remaining samples from the crossfade optimizer.
            if let Some(ref mut optimizer) = self.crossfade_optimizer {
                if let Some(remainder) = optimizer.flush() {
                    if !remainder.is_empty() {
                        let audio_chunk = assembler.push_raw(remainder)?;
                        assembled_chunks.push(audio_chunk);
                    }
                }
            }

            Ok(assembled_chunks)
        })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        // Reset the crossfade optimizer for the next synthesis call.
        if let Some(ref mut optimizer) = self.crossfade_optimizer {
            optimizer.reset();
        }

        result
    }

    /// Streaming chorus with per-voice inputs (different text per voice).
    ///
    /// Each voice synthesizes its own chunk sequence. All voices must have
    /// the same number of chunks (chunk `i` from each voice is mixed together).
    ///
    /// # Arguments
    ///
    /// * `per_voice_chunks` - Outer: voices, inner: chunks per voice.
    ///   All voices must have the same number of chunks.
    /// * `styles` - Per-voice style embeddings. Length must equal `n_voices`.
    /// * `speed` - Speaking rate multiplier (shared).
    /// * `stream_config` - Crossfade configuration.
    /// * `cache` - Metal pipeline cache.
    pub fn synthesize_streaming_chorus_per_voice(
        &mut self,
        per_voice_chunks: &[Vec<DynTensor>],
        styles: &[DynTensor],
        speed: f32,
        stream_config: &KokoroStreamConfig,
        cache: &PipelineCache,
    ) -> Result<Vec<AudioChunk>, CompiledKokoroError> {
        let n = self.voices.len();
        if per_voice_chunks.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "per_voice_chunks length {} != n_voices {n}",
                per_voice_chunks.len()
            ))
            .into());
        }
        if styles.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            ))
            .into());
        }

        // Verify all voices have the same chunk count.
        if n > 0 {
            let expected_chunks = per_voice_chunks[0].len();
            for (vi, vc) in per_voice_chunks.iter().enumerate() {
                if vc.len() != expected_chunks {
                    return Err(KokoroError::InvalidInput(format!(
                        "voice {vi} has {} chunks, expected {expected_chunks}",
                        vc.len()
                    ))
                    .into());
                }
            }
            if expected_chunks == 0 {
                return Ok(Vec::new());
            }
        }
        validate_speed(speed)?;
        let max_position_embeddings = self.voices[0].config().plbert.max_position_embeddings;
        for (vi, voice_chunks) in per_voice_chunks.iter().enumerate() {
            for (ci, chunk_input) in voice_chunks.iter().enumerate() {
                validate_input_ids(chunk_input, max_position_embeddings).map_err(|err| {
                    contextualize_invalid_input(err, format!("per_voice_chunks[{vi}][{ci}]"))
                })?;
            }
        }

        // Reset SineGen phase on all voices at the start of a new streaming
        // session so each voice begins with zero phase.
        for voice in &mut self.voices {
            voice.reset_sinegen_phase();
        }

        // Reclaim buffer pool entries from previous synthesis call.
        crate::arena::pool_reclaim();

        let result = (|| -> Result<Vec<AudioChunk>, CompiledKokoroError> {
            let expected_chunks = per_voice_chunks[0].len();

            // Determine channel count for crossfade assembler.
            let is_stereo = self.config.pans.is_some()
                || self.has_stereo()
                || self.has_chorus_pipeline()
                || self.has_spatial();
            let channels: usize = if is_stereo { 2 } else { 1 };
            let effective_config = if is_stereo {
                KokoroStreamConfig::new(stream_config.crossfade_samples * 2)?
                    .with_window(stream_config.crossfade_window)
            } else {
                stream_config.clone()
            };
            let mut assembler =
                StreamingAssembler::new_with_channels(effective_config, expected_chunks, channels)?;
            let mut assembled_chunks: Vec<AudioChunk> = Vec::with_capacity(expected_chunks);

            // Process chunk-by-chunk across all voices so each chunk can go
            // through the full processing pipeline (mix_or_process) with
            // persistent state across chunk boundaries.
            for ci in 0..expected_chunks {
                // Synthesize this chunk for all voices independently.
                let mut chunk_pcms: Vec<Vec<f32>> = Vec::with_capacity(n);
                for (vi, voice) in self.voices.iter_mut().enumerate() {
                    let (audio_tensor, _cert) =
                        voice.synthesize(&per_voice_chunks[vi][ci], &styles[vi], speed, cache)?;
                    let pcm = audio_tensor
                        .to_flat_vec::<f32>()
                        .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
                    chunk_pcms.push(pcm);
                }

                // Route through the full processing chain (same as
                // non-streaming path). Processing state persists on `self`.
                let mixed = self.mix_or_process(&mut chunk_pcms)?;

                // Optional crossfade boundary optimization.
                let optimized = if let Some(ref mut optimizer) = self.crossfade_optimizer {
                    optimizer.push_chunk(&mixed).map_err(|e| {
                        CompiledKokoroError::InvalidInput(format!("crossfade optimizer: {e}"))
                    })?
                } else {
                    mixed
                };

                // Crossfade via StreamingAssembler.
                let audio_chunk = assembler.push_raw(optimized)?;
                assembled_chunks.push(audio_chunk);
            }

            // Flush any remaining samples from the crossfade optimizer.
            if let Some(ref mut optimizer) = self.crossfade_optimizer {
                if let Some(remainder) = optimizer.flush() {
                    if !remainder.is_empty() {
                        let audio_chunk = assembler.push_raw(remainder)?;
                        assembled_chunks.push(audio_chunk);
                    }
                }
            }

            Ok(assembled_chunks)
        })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        // Reset the crossfade optimizer for the next synthesis call.
        if let Some(ref mut optimizer) = self.crossfade_optimizer {
            optimizer.reset();
        }

        result
    }

    /// Streaming chorus with persistent reverb (shared text, all voices same chunks).
    ///
    /// Identical to [`synthesize_streaming_chorus`](Self::synthesize_streaming_chorus)
    /// but applies a persistent [`StreamingReverb`] across all output chunks after
    /// mixing and crossfade. The reverb delay-line state carries from one chunk to
    /// the next, producing a smooth, continuous reverb tail across chunk boundaries.
    ///
    /// Use this instead of setting `ChorusConfig.reverb` when you want the reverb
    /// tail to persist across chunks. `ChorusConfig.reverb` applies reverb
    /// independently per chunk (state is lost at each boundary), while this method
    /// uses [`StreamingReverb`] to maintain comb/allpass filter state across the
    /// full sequence.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-chunked token IDs as `DynTensor`s, each `[1, T_i]`.
    /// * `styles` - Per-voice style embeddings. Length must equal `n_voices`.
    /// * `speed` - Speaking rate multiplier (shared).
    /// * `stream_config` - Crossfade configuration.
    /// * `reverb_config` - Reverb parameters. Use preset constructors like
    ///   [`ReverbConfig::medium_hall()`] or build a custom config.
    /// * `cache` - Metal pipeline cache.
    pub fn synthesize_streaming_chorus_with_reverb(
        &mut self,
        chunks: &[DynTensor],
        styles: &[DynTensor],
        speed: f32,
        stream_config: &KokoroStreamConfig,
        reverb_config: &ReverbConfig,
        cache: &PipelineCache,
    ) -> Result<Vec<AudioChunk>, CompiledKokoroError> {
        let mut audio_chunks =
            self.synthesize_streaming_chorus(chunks, styles, speed, stream_config, cache)?;
        apply_streaming_reverb_to_chunks(&mut audio_chunks, reverb_config, &self.config)?;
        Ok(audio_chunks)
    }

    /// Streaming chorus with persistent reverb (per-voice inputs).
    ///
    /// Identical to
    /// [`synthesize_streaming_chorus_per_voice`](Self::synthesize_streaming_chorus_per_voice)
    /// but applies a persistent [`StreamingReverb`] across all output chunks.
    /// See [`synthesize_streaming_chorus_with_reverb`](Self::synthesize_streaming_chorus_with_reverb)
    /// for details on streaming vs. batch reverb.
    ///
    /// # Arguments
    ///
    /// * `per_voice_chunks` - Outer: voices, inner: chunks per voice.
    /// * `styles` - Per-voice style embeddings. Length must equal `n_voices`.
    /// * `speed` - Speaking rate multiplier (shared).
    /// * `stream_config` - Crossfade configuration.
    /// * `reverb_config` - Reverb parameters.
    /// * `cache` - Metal pipeline cache.
    pub fn synthesize_streaming_chorus_per_voice_with_reverb(
        &mut self,
        per_voice_chunks: &[Vec<DynTensor>],
        styles: &[DynTensor],
        speed: f32,
        stream_config: &KokoroStreamConfig,
        reverb_config: &ReverbConfig,
        cache: &PipelineCache,
    ) -> Result<Vec<AudioChunk>, CompiledKokoroError> {
        let mut audio_chunks = self.synthesize_streaming_chorus_per_voice(
            per_voice_chunks,
            styles,
            speed,
            stream_config,
            cache,
        )?;
        apply_streaming_reverb_to_chunks(&mut audio_chunks, reverb_config, &self.config)?;
        Ok(audio_chunks)
    }
}

/// Apply persistent streaming reverb to a sequence of assembled audio chunks.
///
/// Creates a [`StreamingReverb`] and processes all chunks in order so that the
/// reverb delay-line state (comb and allpass filters) carries from one chunk to
/// the next. This eliminates the discontinuities that occur when reverb is
/// applied independently per chunk.
///
/// The `chorus_config` is inspected to determine whether the output is stereo
/// (interleaved L/R samples) based on whether `pans` is set.
fn apply_streaming_reverb_to_chunks(
    chunks: &mut [AudioChunk],
    reverb_config: &ReverbConfig,
    chorus_config: &nn_models::kokoro_chorus::ChorusConfig,
) -> Result<(), CompiledKokoroError> {
    let is_stereo = chorus_config.pans.is_some();
    let mut reverb = StreamingReverb::new(reverb_config.clone(), is_stereo)?;
    reverb.apply_to_chunks(chunks);
    Ok(())
}
