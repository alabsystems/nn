// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Incremental crossfade assembler for streaming Kokoro synthesis.
//!
//! Extracted from `kokoro_streaming.rs` to comply with the 500-line limit.
//! Re-exported via `kokoro_streaming::StreamingAssembler`.
//!
//! Part of #3496, #3351 (T4.1).

use super::{crossfade_blend_into_windowed, AudioChunk, KokoroStreamConfig};
use crate::kokoro_error::KokoroError;

/// Incremental crossfade assembler for streaming synthesis.
///
/// Unlike [`super::assemble_streaming_chunks`] which requires all raw PCM upfront,
/// `StreamingAssembler` processes one chunk at a time. This enables
/// sub-utterance playback latency: the caller can start playing audio from
/// the first chunk while subsequent chunks are still being synthesized.
///
/// # Usage
///
/// ```rust,ignore
/// let mut assembler = StreamingAssembler::new(config, num_chunks)?;
/// for (i, chunk_input) in token_chunks.iter().enumerate() {
///     let raw_pcm = kokoro.synthesize(chunk_input, &style, speed, &cache)?;
///     let audio_chunk = assembler.push_raw(raw_pcm)?;
///     play_audio(&audio_chunk.pcm);  // immediate playback
/// }
/// assert!(assembler.is_complete());
/// ```
///
/// # Memory
///
/// Only retains the crossfade tail of the previous chunk (up to
/// `crossfade_samples` floats) — memory usage is O(crossfade_samples),
/// not O(total_audio).
///
/// Part of #3351 (T4.1).
#[derive(Debug)]
pub struct StreamingAssembler {
    config: KokoroStreamConfig,
    total_chunks: usize,
    /// Number of audio channels (1 = mono, 2 = interleaved stereo).
    /// The effective crossfade float count is `crossfade_samples * channels`
    /// because interleaved stereo has 2 floats per time-domain sample.
    channels: usize,
    /// Tail of the previous raw chunk for crossfade blending.
    prev_tail: Option<Vec<f32>>,
    /// Running sample offset for the next chunk.
    sample_offset: usize,
    /// Index of the next chunk to be pushed.
    next_index: usize,
}

impl StreamingAssembler {
    /// Create a new mono assembler for `total_chunks` sequential raw PCM pushes.
    ///
    /// Equivalent to `new_with_channels(config, total_chunks, 1)`.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails or
    /// `total_chunks` is 0.
    pub fn new(config: KokoroStreamConfig, total_chunks: usize) -> Result<Self, KokoroError> {
        Self::new_with_channels(config, total_chunks, 1)
    }

    /// Create a new assembler for `total_chunks` sequential raw PCM pushes
    /// with `channels` audio channels (1 = mono, 2 = interleaved stereo).
    ///
    /// For multi-channel audio, the effective crossfade float count is
    /// `crossfade_samples * channels` because interleaved stereo encodes
    /// 2 floats per time-domain sample.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if config validation fails,
    /// `total_chunks` is 0, or `channels` is 0.
    pub fn new_with_channels(
        config: KokoroStreamConfig,
        total_chunks: usize,
        channels: usize,
    ) -> Result<Self, KokoroError> {
        config.validate()?;
        if total_chunks == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "total_chunks",
                reason: "must be > 0".into(),
            });
        }
        if channels == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "channels",
                reason: "must be > 0".into(),
            });
        }
        Ok(Self {
            config,
            total_chunks,
            channels,
            prev_tail: None,
            sample_offset: 0,
            next_index: 0,
        })
    }

    /// Push a raw PCM chunk and get back the assembled [`AudioChunk`].
    ///
    /// Applies crossfade with the previous chunk's tail (if any) and returns
    /// an `AudioChunk` ready for immediate playback.
    ///
    /// Must be called exactly `total_chunks` times, in order.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Called after all chunks have been pushed (`is_complete()` is true)
    /// - The raw PCM is shorter than `crossfade_samples`
    pub fn push_raw(&mut self, raw_pcm: Vec<f32>) -> Result<AudioChunk, KokoroError> {
        if self.next_index >= self.total_chunks {
            return Err(KokoroError::InvalidInput(format!(
                "all {} chunks already pushed",
                self.total_chunks,
            )));
        }

        // Effective crossfade float count: for stereo interleaved audio,
        // each time-domain sample occupies `channels` floats.
        let cf = self.config.crossfade_samples * self.channels;
        let is_first = self.next_index == 0;
        let is_last = self.next_index == self.total_chunks - 1;
        let chunk_index = self.next_index;

        // Non-last chunks must be at least crossfade_samples long to provide
        // a valid tail for the next chunk's crossfade blend.
        if !is_last && self.total_chunks > 1 && raw_pcm.len() < cf {
            return Err(KokoroError::InvalidInput(format!(
                "chunk {} too short for crossfade: {} < {} \
                 (non-last chunks must be >= crossfade_samples)",
                chunk_index,
                raw_pcm.len(),
                cf,
            )));
        }

        // For non-last chunks, hold back the crossfade tail.
        let emit_len = if is_last {
            raw_pcm.len()
        } else {
            raw_pcm.len().saturating_sub(cf)
        };

        let pcm = if is_first {
            // No crossfade — emit the playable portion.
            raw_pcm[..emit_len].to_vec()
        } else {
            // Crossfade with previous chunk's tail.
            let prev_tail = self.prev_tail.as_ref().ok_or_else(|| {
                KokoroError::InvalidInput("missing previous tail for crossfade".into())
            })?;

            if raw_pcm.len() < cf {
                return Err(KokoroError::InvalidInput(format!(
                    "chunk {} too short for crossfade: {} < {}",
                    chunk_index,
                    raw_pcm.len(),
                    cf,
                )));
            }

            // Defense-in-depth: prev_tail should always be >= cf (the early
            // validation above ensures non-last chunks are long enough), but
            // guard against index-out-of-bounds in crossfade_blend_into.
            if prev_tail.len() < cf {
                return Err(KokoroError::InvalidInput(format!(
                    "previous chunk tail too short for crossfade: {} < {} \
                     (chunk {} was shorter than crossfade_samples)",
                    prev_tail.len(),
                    cf,
                    chunk_index - 1,
                )));
            }

            let mut out = Vec::with_capacity(emit_len);
            crossfade_blend_into_windowed(
                &mut out,
                prev_tail,
                &raw_pcm,
                cf,
                emit_len,
                self.config.crossfade_window,
            );
            // Copy remaining non-crossfade samples up to emit_len.
            if emit_len > cf {
                out.extend_from_slice(&raw_pcm[cf..emit_len]);
            }
            out
        };

        // Save the crossfade tail for the next chunk (unless this is the last).
        if is_last {
            self.prev_tail = None;
        } else if raw_pcm.len() >= cf {
            self.prev_tail = Some(raw_pcm[raw_pcm.len() - cf..].to_vec());
        } else {
            // Chunk shorter than crossfade — save entire chunk as tail.
            self.prev_tail = Some(raw_pcm);
        }

        let chunk = AudioChunk {
            pcm,
            channels: self.channels,
            sample_offset: self.sample_offset,
            chunk_index,
            total_chunks: self.total_chunks,
            is_final: is_last,
        };

        self.sample_offset += chunk.pcm.len();
        self.next_index += 1;

        Ok(chunk)
    }

    /// Whether all chunks have been pushed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.next_index >= self.total_chunks
    }

    /// Number of chunks remaining to be pushed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.total_chunks.saturating_sub(self.next_index)
    }

    /// Index of the next chunk to be pushed (0-based).
    #[must_use]
    pub fn next_index(&self) -> usize {
        self.next_index
    }

    /// Total number of chunks this assembler expects.
    #[must_use]
    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    /// Running sample offset (total PCM floats emitted so far).
    #[must_use]
    pub fn sample_offset(&self) -> usize {
        self.sample_offset
    }

    /// Reset the assembler to accept chunks from the beginning again.
    ///
    /// Clears crossfade tail, resets cursor and sample offset. Useful for
    /// re-synthesizing the same text without allocating a new assembler.
    pub fn reset(&mut self) {
        self.prev_tail = None;
        self.sample_offset = 0;
        self.next_index = 0;
    }
}
