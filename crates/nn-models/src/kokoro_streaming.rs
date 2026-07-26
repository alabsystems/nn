// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming synthesis API contract for Kokoro TTS.
//!
//! Defines the types and crossfade logic shared between nn's Kokoro synthesis
//! and dvoice's chorus conductor. The streaming contract enables:
//!
//! - **Chunked synthesis:** Long text split into ≤512-token chunks, each
//!   synthesized independently.
//! - **Crossfade blending:** Adjacent chunks overlap by `crossfade_samples`
//!   with linear crossfade to eliminate boundary clicks.
//! - **Incremental delivery:** Each [`AudioChunk`] is playable as soon as
//!   synthesized — no need to wait for the full utterance.
//!
//! # Architecture
//!
//! ```text
//! KokoroTokenizer::chunk_and_encode()
//!     → Vec<(String, Vec<u32>)>       // phoneme chunks + token IDs
//!     → synthesize_streaming()         // this module
//!     → Vec<AudioChunk>               // crossfaded PCM chunks
//!     → dvoice conductor_choir.rs   // playback
//! ```
//!
//! Two implementations exist:
//! - **CPU path:** [`KokoroModel::synthesize_streaming`] (this module, nn-models)
//! - **GPU path:** `CompiledKokoro::synthesize_streaming` (nn-metal)
//!
//! Part of #3355, #3351, #2918.

use crate::kokoro_chorus::{mix_voices_from_refs, ChorusConfig};
use crate::kokoro_error::{validate_speed, KokoroError};
use crate::kokoro_tts::KokoroModel;
use nn_core::dyn_tensor::DynTensor;

#[path = "kokoro_streaming_types.rs"]
mod streaming_types;
pub use streaming_types::{
    concatenate_chunks, AudioChunk, CrossfadeWindow, KokoroStreamConfig, HANN_CROSSFADE_THRESHOLD,
};

// ---------------------------------------------------------------------------
// Crossfade
// ---------------------------------------------------------------------------

/// Apply crossfade between the tail of the previous chunk and the head of
/// the next chunk, using the specified window function.
///
/// Modifies `next_pcm` in place: the first `crossfade_samples` are blended
/// with the last `crossfade_samples` of `prev_pcm`.
///
/// Blend formula: `out[i] = prev[i] * (1 - alpha) + next[i] * alpha`
/// where alpha depends on the window:
/// - **Linear**: `alpha = i / (crossfade_samples - 1)`
/// - **Hann**: `alpha = 0.5 * (1 - cos(PI * i / (crossfade_samples - 1)))`
/// - **SqrtHann**: `alpha = sqrt(0.5 * (1 - cos(PI * i / (crossfade_samples - 1))))`
///
/// Delegates to [`nn_core::audio::crossfade_linear_blend`] or
/// [`nn_core::audio::crossfade_hann_blend`] for the core blend math.
///
/// # Errors
///
/// Returns `KokoroError::InvalidInput` if either chunk is shorter than
/// `crossfade_samples`.
pub fn crossfade_chunks(
    prev_pcm: &[f32],
    next_pcm: &mut [f32],
    crossfade_samples: usize,
) -> Result<(), KokoroError> {
    crossfade_chunks_windowed(
        prev_pcm,
        next_pcm,
        crossfade_samples,
        CrossfadeWindow::Linear,
    )
}

/// Apply crossfade with explicit window selection.
///
/// See [`crossfade_chunks`] for details.
pub fn crossfade_chunks_windowed(
    prev_pcm: &[f32],
    next_pcm: &mut [f32],
    crossfade_samples: usize,
    window: CrossfadeWindow,
) -> Result<(), KokoroError> {
    if crossfade_samples == 0 {
        return Ok(());
    }
    if prev_pcm.len() < crossfade_samples {
        return Err(KokoroError::InvalidInput(format!(
            "previous chunk too short for crossfade: {} < {}",
            prev_pcm.len(),
            crossfade_samples,
        )));
    }
    if next_pcm.len() < crossfade_samples {
        return Err(KokoroError::InvalidInput(format!(
            "next chunk too short for crossfade: {} < {}",
            next_pcm.len(),
            crossfade_samples,
        )));
    }

    // Delegate to the canonical crossfade in nn-core, then copy blended
    // samples back into next_pcm in-place.
    let tail = &prev_pcm[prev_pcm.len() - crossfade_samples..];
    let head = &next_pcm[..crossfade_samples];
    let blended = match window {
        CrossfadeWindow::SqrtHann => {
            nn_core::audio::crossfade_sqrt_hann_blend(tail, head, crossfade_samples)
        }
        CrossfadeWindow::Hann => {
            nn_core::audio::crossfade_hann_blend(tail, head, crossfade_samples)
        }
        // Linear or any future variants fall back to linear crossfade.
        _ => nn_core::audio::crossfade_linear_blend(tail, head, crossfade_samples),
    };
    next_pcm[..crossfade_samples].copy_from_slice(&blended);
    Ok(())
}

/// Blend `tail` and `head` using the specified crossfade window, appending
/// results to `out`.
///
/// `cf` = crossfade sample count, `limit` = max samples to emit (may be < cf
/// for very short chunks), `window` = crossfade curve.
///
/// Delegates to [`nn_core::audio::crossfade_blend_into`] (linear) or
/// [`nn_core::audio::crossfade_hann_blend_into`] (Hann).
///
/// Kani coverage: `crossfade_alpha_in_unit_interval` (harness 1) and
/// `crossfade_convex_combination_bounded` (harness 3) prove the alpha and
/// output bounds for the linear formula.
/// Retained for backward compatibility with Kani harnesses that call the
/// non-windowed variant. Production code should use `crossfade_blend_into_windowed`.
#[allow(dead_code)]
pub(crate) fn crossfade_blend_into(
    out: &mut Vec<f32>,
    tail: &[f32],
    head: &[f32],
    cf: usize,
    limit: usize,
) {
    crossfade_blend_into_windowed(out, tail, head, cf, limit, CrossfadeWindow::Linear);
}

/// Blend with explicit window selection. See [`crossfade_blend_into`].
pub(crate) fn crossfade_blend_into_windowed(
    out: &mut Vec<f32>,
    tail: &[f32],
    head: &[f32],
    cf: usize,
    limit: usize,
    window: CrossfadeWindow,
) {
    match window {
        CrossfadeWindow::SqrtHann => {
            nn_core::audio::crossfade_sqrt_hann_blend_into(out, tail, head, cf, limit);
        }
        CrossfadeWindow::Hann => {
            nn_core::audio::crossfade_hann_blend_into(out, tail, head, cf, limit);
        }
        // Linear or any future variants fall back to linear crossfade.
        _ => {
            nn_core::audio::crossfade_blend_into(out, tail, head, cf, limit);
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming synthesis (assembles AudioChunks from raw PCM slices)
// ---------------------------------------------------------------------------

/// Assemble a sequence of raw PCM chunks into crossfaded [`AudioChunk`]s.
///
/// This is the shared assembly logic used by both CPU (`KokoroModel`) and
/// GPU (`CompiledKokoro`) streaming paths. Each backend synthesizes raw PCM
/// for each token chunk, then calls this function to apply crossfade and
/// produce the final `AudioChunk` sequence.
///
/// # Arguments
///
/// * `raw_chunks` - Raw PCM audio for each synthesized chunk, in order.
/// * `config` - Streaming configuration (crossfade length).
///
/// # Returns
///
/// A `Vec<AudioChunk>` with crossfade applied at all boundaries.
/// Each chunk's `pcm` is ready for immediate playback.
///
/// # Errors
///
/// Returns `KokoroError::InvalidInput` if any chunk is shorter than
/// `crossfade_samples`.
pub fn assemble_streaming_chunks(
    raw_chunks: &[Vec<f32>],
    config: &KokoroStreamConfig,
) -> Result<Vec<AudioChunk>, KokoroError> {
    config.validate()?;

    if raw_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let total_chunks = raw_chunks.len();
    let cf = config.crossfade_samples;

    if total_chunks == 1 {
        return Ok(vec![AudioChunk {
            pcm: raw_chunks[0].clone(),
            channels: 1,
            sample_offset: 0,
            chunk_index: 0,
            total_chunks: 1,
            is_final: true,
        }]);
    }

    let mut result = Vec::with_capacity(total_chunks);
    let mut sample_offset: usize = 0;

    for (i, raw) in raw_chunks.iter().enumerate() {
        let is_first = i == 0;
        let is_last = i == total_chunks - 1;

        // Emit length: non-last chunks exclude the crossfade tail
        // (it will be blended into the next chunk's head).
        let emit_len = if is_last {
            raw.len()
        } else {
            raw.len().saturating_sub(cf)
        };

        // Build output directly without cloning the full raw chunk.
        // For non-first chunks, blend the crossfade region inline.
        let pcm = if is_first {
            // No crossfade — just copy the emitted portion.
            raw[..emit_len].to_vec()
        } else {
            let prev = &raw_chunks[i - 1];
            // Validate lengths for crossfade.
            if prev.len() < cf {
                return Err(KokoroError::InvalidInput(format!(
                    "previous chunk too short for crossfade: {} < {}",
                    prev.len(),
                    cf,
                )));
            }
            if raw.len() < cf {
                return Err(KokoroError::InvalidInput(format!(
                    "next chunk too short for crossfade: {} < {}",
                    raw.len(),
                    cf,
                )));
            }
            let tail = &prev[prev.len() - cf..];
            let mut out = Vec::with_capacity(emit_len);
            crossfade_blend_into_windowed(
                &mut out,
                tail,
                raw,
                cf,
                emit_len,
                config.crossfade_window,
            );
            // Copy remaining non-crossfade samples up to emit_len.
            if emit_len > cf {
                out.extend_from_slice(&raw[cf..emit_len]);
            }
            out
        };

        result.push(AudioChunk {
            pcm,
            channels: 1,
            sample_offset,
            chunk_index: i,
            total_chunks,
            is_final: is_last,
        });

        sample_offset += emit_len;
    }

    Ok(result)
}

/// Assemble owned raw PCM chunks into crossfaded [`AudioChunk`]s with minimal
/// copying.
///
/// Unlike [`assemble_streaming_chunks`], this helper consumes `raw_chunks` and
/// reuses each chunk's allocation for the emitted [`AudioChunk`] wherever
/// possible:
///
/// - the first chunk is truncated in place instead of copied,
/// - non-first chunks have their crossfade head blended in place,
/// - the final chunk is emitted without rebuilding its PCM buffer.
///
/// This keeps streaming chorus and CPU streaming on the owned-data path from
/// cloning or rebuilding chunk PCM that is already materialized.
fn assemble_streaming_chunks_owned(
    raw_chunks: Vec<Vec<f32>>,
    config: &KokoroStreamConfig,
) -> Result<Vec<AudioChunk>, KokoroError> {
    config.validate()?;

    let total_chunks = raw_chunks.len();
    if total_chunks == 0 {
        return Ok(Vec::new());
    }

    let cf = config.crossfade_samples;
    let mut result = Vec::with_capacity(total_chunks);
    let mut sample_offset: usize = 0;
    let mut iter = raw_chunks.into_iter().enumerate();
    let (mut prev_idx, mut prev_raw) = iter
        .next()
        .expect("owned streaming assembly already checked for empty input");

    for (chunk_index, mut raw) in iter {
        if cf > 0 {
            crossfade_chunks_windowed(&prev_raw, raw.as_mut_slice(), cf, config.crossfade_window)?;
        }

        let emit_len = prev_raw.len().saturating_sub(cf);
        prev_raw.truncate(emit_len);
        result.push(AudioChunk {
            pcm: prev_raw,
            channels: 1,
            sample_offset,
            chunk_index: prev_idx,
            total_chunks,
            is_final: false,
        });
        sample_offset += emit_len;

        prev_idx = chunk_index;
        prev_raw = raw;
    }

    result.push(AudioChunk {
        pcm: prev_raw,
        channels: 1,
        sample_offset,
        chunk_index: prev_idx,
        total_chunks,
        is_final: true,
    });

    Ok(result)
}

// ---------------------------------------------------------------------------
// Streaming chorus assembly (multi-voice + chunked crossfade)
// ---------------------------------------------------------------------------

/// Assemble a streaming chorus: mix per-chunk across voices, then crossfade
/// between adjacent mixed chunks.
///
/// This is the backend-agnostic assembly logic for combined streaming + chorus.
/// Each voice produces raw PCM for each text chunk. This function:
///
/// 1. For each chunk index, mixes all voices' audio via [`mix_voices_from_refs`].
/// 2. Applies linear crossfade between adjacent mixed chunks.
/// 3. Returns `Vec<AudioChunk>` ready for incremental playback.
///
/// When `chorus_config` has stereo pans set, produces interleaved stereo output
/// and doubles the crossfade overlap to preserve the same time duration.
///
/// First-audio latency is 1 chunk × N voices (not full utterance × N voices).
///
/// # Arguments
///
/// * `per_voice_chunks` - Outer: voices (length = N), inner: chunks per voice.
///   All voices must have the same number of chunks. Chunk `i` from each voice
///   corresponds to the same text segment.
/// * `chorus_config` - Voice count, gains, optional stereo pans, clipping.
/// * `stream_config` - Crossfade configuration for chunk boundaries.
///
/// # Errors
///
/// Returns error if voice counts don't match, chunk counts are inconsistent,
/// or any chunk is too short for crossfade.
pub fn assemble_streaming_chorus(
    per_voice_chunks: &[Vec<Vec<f32>>],
    chorus_config: &ChorusConfig,
    stream_config: &KokoroStreamConfig,
) -> Result<Vec<AudioChunk>, KokoroError> {
    chorus_config.validate()?;
    stream_config.validate()?;

    if per_voice_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let n_voices = per_voice_chunks.len();
    if n_voices != chorus_config.n_voices {
        return Err(KokoroError::InvalidInput(format!(
            "per_voice_chunks length {} != chorus_config.n_voices {}",
            n_voices, chorus_config.n_voices,
        )));
    }

    let n_chunks = per_voice_chunks[0].len();
    if n_chunks == 0 {
        return Ok(Vec::new());
    }

    // Verify all voices have the same number of chunks.
    for (v, voice_chunks) in per_voice_chunks.iter().enumerate() {
        if voice_chunks.len() != n_chunks {
            return Err(KokoroError::InvalidInput(format!(
                "voice {v} has {} chunks, expected {n_chunks}",
                voice_chunks.len(),
            )));
        }
    }

    let is_stereo = chorus_config.pans.is_some();
    let channels: usize = if is_stereo { 2 } else { 1 };

    // Mix each chunk across all voices via mix_voices_from_refs.
    // Uses borrowed slices to avoid cloning full PCM Vec per voice per chunk.
    let mut mixed_raw: Vec<Vec<f32>> = Vec::with_capacity(n_chunks);
    for chunk_idx in 0..n_chunks {
        let voice_slices: Vec<&[f32]> = per_voice_chunks
            .iter()
            .map(|v| v[chunk_idx].as_slice())
            .collect();

        let mixed = mix_voices_from_refs(&voice_slices, chorus_config)?;
        mixed_raw.push(mixed);
    }

    // For stereo, crossfade operates on the flat interleaved float array.
    // Double the crossfade_samples so the crossfade covers the same time
    // duration (crossfade_samples counts per-channel samples, but the
    // interleaved buffer has 2× the floats per time unit).
    let effective_config = if is_stereo {
        KokoroStreamConfig {
            crossfade_samples: stream_config.crossfade_samples * 2,
            crossfade_window: stream_config.crossfade_window,
        }
    } else {
        stream_config.clone()
    };

    // Assemble with crossfade, then stamp the channel count.
    let mut chunks = assemble_streaming_chunks_owned(mixed_raw, &effective_config)?;
    for chunk in &mut chunks {
        chunk.channels = channels;
    }
    Ok(chunks)
}

// ---------------------------------------------------------------------------
// CPU streaming synthesis on KokoroModel
// ---------------------------------------------------------------------------

impl KokoroModel {
    /// Synthesize chunked token sequences with crossfade (CPU path).
    ///
    /// Each `input_ids` tensor is synthesized independently via `forward_audio()`,
    /// then adjacent chunks are crossfaded using the streaming config's overlap.
    ///
    /// This is the CPU analog of `CompiledKokoro::synthesize_streaming()` in
    /// nn-metal. Use the GPU path for production; this CPU path is useful for
    /// testing and environments without Metal.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-chunked token IDs as `DynTensor`s, each `[1, T_i]`.
    ///   Use `KokoroTokenizer::chunk_and_encode()` to split long text.
    /// * `style` - Style embedding `[1, 2*style_dim]` (shared across all chunks).
    /// * `speed` - Speaking rate multiplier (shared across all chunks).
    /// * `stream_config` - Crossfade configuration.
    ///
    /// # Returns
    ///
    /// A `Vec<AudioChunk>` with crossfade applied at all boundaries.
    /// Each chunk's `pcm` is ready for immediate playback.
    pub fn synthesize_streaming(
        &self,
        chunks: &[DynTensor],
        style: &DynTensor,
        speed: f32,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, KokoroError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        validate_speed(speed)?;

        // Reset SineGen phase for this new streaming session. Within the loop,
        // SineGen carries the terminal cumulative phase from each chunk to the
        // next, ensuring phase continuity at chunk boundaries.
        self.reset_sinegen_phase();

        // Synthesize each chunk with SineGen phase continuity via forward_audio.
        let mut raw_chunks: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        for chunk_input in chunks {
            let audio_tensor = self.forward_audio(chunk_input, style, speed)?;
            // forward_audio returns [1, 1, T_audio] — flatten to 1D PCM.
            let pcm = audio_tensor.to_flat_vec::<f32>()?;
            raw_chunks.push(pcm);
        }

        // Assemble with crossfade.
        assemble_streaming_chunks_owned(raw_chunks, stream_config)
    }
}

// ---------------------------------------------------------------------------
// Incremental assembler (sub-utterance latency)
// ---------------------------------------------------------------------------

#[path = "kokoro_streaming_assembler.rs"]
mod assembler;
pub use assembler::StreamingAssembler;

#[path = "kokoro_streaming_session.rs"]
mod streaming_session;
pub use streaming_session::StreamingKokoroSession;

#[cfg(kani)]
#[path = "kokoro_streaming_kani_crossfade.rs"]
mod kani_crossfade;

#[cfg(kani)]
#[path = "kokoro_streaming_kani_contiguity.rs"]
mod kani_contiguity;

#[cfg(kani)]
#[path = "kokoro_streaming_kani_assembler.rs"]
mod kani_assembler;

#[cfg(kani)]
#[path = "kani_kokoro_streaming.rs"]
mod kani_streaming_proofs;

#[cfg(test)]
#[path = "kokoro_streaming_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kokoro_streaming_assembler_tests.rs"]
mod assembler_tests;

#[cfg(test)]
#[path = "kokoro_streaming_chorus_tests.rs"]
mod chorus_tests;

#[cfg(test)]
#[path = "kokoro_streaming_assembly_proofs.rs"]
mod assembly_proofs;

#[cfg(test)]
#[path = "kokoro_streaming_session_tests.rs"]
mod session_tests;
