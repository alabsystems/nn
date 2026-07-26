// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming chorus session for real-time Kokoro TTS.
//!
//! [`StreamingChorusSession`] wraps [`ChorusMasterPipeline`] to process audio
//! in small chunks (e.g. 256--1024 samples) while maintaining all DSP state
//! (reverb tails, compressor envelopes, filter memories, delay lines) across
//! chunk boundaries.
//!
//! # Usage
//!
//! ```text
//! let config = ChorusMasterConfig::singing_chorus(4)?;
//! let mut session = StreamingChorusSession::new(config)?;
//!
//! // Process chunks from 4 voices:
//! for chunk_idx in 0..num_chunks {
//!     let voice_chunks: Vec<&[f32]> = voices.iter()
//!         .map(|v| &v[chunk_idx * 512..(chunk_idx + 1) * 512])
//!         .collect();
//!     let mut left = vec![0.0f32; 512];
//!     let mut right = vec![0.0f32; 512];
//!     session.process_chunk(&voice_chunks, &mut left, &mut right)?;
//!     // left/right are ready for playback
//! }
//!
//! // Drain reverb/delay tails at end of utterance:
//! let (tail_l, tail_r) = session.flush()?;
//! ```
//!
//! # Design
//!
//! The streaming session delegates entirely to [`ChorusMasterPipeline::process`]
//! for DSP logic. It does NOT duplicate any signal processing -- it only
//! adapts the chunk-based calling convention to the pipeline's buffer-based API.
//! All filter state (IIR memories, delay lines, envelope followers, LFO phases,
//! FFT overlap buffers, PRNG state) lives inside the pipeline and persists
//! between `process_chunk` calls automatically.
//!
//! Part of #3351, #4264.

use crate::kokoro_chorus_pipeline::{ChorusMasterConfig, ChorusMasterPipeline};
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

/// Default number of silent samples to push through the pipeline during
/// [`flush`](StreamingChorusSession::flush) to drain reverb and delay tails.
/// At 24 kHz this is ~170 ms -- enough for typical chorus reverb tails.
const DEFAULT_FLUSH_SAMPLES: usize = 4096;

/// Streaming chorus session for real-time chunk-by-chunk processing.
///
/// Constructed once with a [`ChorusMasterConfig`], then fed audio chunks
/// via [`process_chunk`](Self::process_chunk). All DSP processor state
/// (reverb, compressor, EQ filters, delay lines, etc.) persists between
/// chunks automatically because the underlying [`ChorusMasterPipeline`]
/// is retained across calls.
///
/// # Thread safety
///
/// `StreamingChorusSession` is `Send` but not `Sync` (it holds `&mut`
/// state). Typical usage: one session per audio output thread.
pub struct StreamingChorusSession {
    pipeline: ChorusMasterPipeline,
    n_voices: usize,
    total_samples_processed: u64,
    sample_rate: u32,
}

impl StreamingChorusSession {
    /// Create a new streaming session from the given config.
    ///
    /// Pre-allocates all DSP processor state (filters, delay lines, etc.).
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the config is invalid.
    pub fn new(config: ChorusMasterConfig) -> Result<Self, KokoroError> {
        let n_voices = config.n_voices;
        let pipeline = ChorusMasterPipeline::new(config)?;
        Ok(Self {
            pipeline,
            n_voices,
            total_samples_processed: 0,
            sample_rate: KOKORO_SAMPLE_RATE as u32,
        })
    }

    /// Process one chunk of audio through the full chorus pipeline.
    ///
    /// Each entry in `voices` is a slice of mono PCM f32 samples for one
    /// voice. All slices must have the same length. The processed stereo
    /// output is written into `left` and `right`, which must be at least
    /// as long as the voice slices.
    ///
    /// DSP state from this call carries over to the next `process_chunk`
    /// call (reverb tails, compressor envelopes, filter memories, etc.).
    ///
    /// # Arguments
    ///
    /// * `voices` -- per-voice audio chunks. Length must equal `n_voices`
    ///   from the config.
    /// * `left` -- output buffer for left channel. Must be >= voice length.
    /// * `right` -- output buffer for right channel. Must be >= voice length.
    ///
    /// # Errors
    ///
    /// - `KokoroError::InvalidInput` if `voices.len() != n_voices`
    /// - `KokoroError::InvalidInput` if voice slices have different lengths
    /// - `KokoroError::InvalidInput` if output buffers are too short
    /// - Any error from the underlying pipeline processing
    pub fn process_chunk(
        &mut self,
        voices: &[&[f32]],
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<(), KokoroError> {
        if voices.len() != self.n_voices {
            return Err(KokoroError::InvalidInput(format!(
                "expected {} voices, got {}",
                self.n_voices,
                voices.len(),
            )));
        }

        if voices.is_empty() {
            return Ok(());
        }

        let chunk_len = voices[0].len();
        if chunk_len == 0 {
            return Ok(());
        }

        // Validate all voices have the same length.
        for (i, voice) in voices.iter().enumerate() {
            if voice.len() != chunk_len {
                return Err(KokoroError::InvalidInput(format!(
                    "voice {i} has {} samples, expected {chunk_len} (all voices must match)",
                    voice.len(),
                )));
            }
        }

        // Validate output buffers are large enough.
        if left.len() < chunk_len {
            return Err(KokoroError::InvalidInput(format!(
                "left output buffer too short: {} < {chunk_len}",
                left.len(),
            )));
        }
        if right.len() < chunk_len {
            return Err(KokoroError::InvalidInput(format!(
                "right output buffer too short: {} < {chunk_len}",
                right.len(),
            )));
        }

        // Convert borrowed slices to owned vecs for the pipeline API.
        let owned_voices: Vec<Vec<f32>> = voices.iter().map(|v| v.to_vec()).collect();

        // Process through the full pipeline -- state persists in self.pipeline.
        let (out_left, out_right) = self.pipeline.process(&owned_voices)?;

        // Copy output into caller's buffers.
        let copy_len = chunk_len.min(out_left.len()).min(out_right.len());
        left[..copy_len].copy_from_slice(&out_left[..copy_len]);
        right[..copy_len].copy_from_slice(&out_right[..copy_len]);

        // Zero any remaining output buffer space (should not happen normally,
        // but defensive against pipeline producing fewer samples than input).
        for s in left[copy_len..chunk_len].iter_mut() {
            *s = 0.0;
        }
        for s in right[copy_len..chunk_len].iter_mut() {
            *s = 0.0;
        }

        self.total_samples_processed += chunk_len as u64;
        Ok(())
    }

    /// Flush remaining DSP tails (reverb, delay, compressor release).
    ///
    /// Pushes silent input through the pipeline to drain any trailing
    /// energy from stateful processors. Returns the stereo tail audio.
    /// Call this at the end of an utterance before starting a new one.
    ///
    /// After flush, call [`reset`](Self::reset) to prepare for a new
    /// utterance, or simply drop the session.
    ///
    /// # Returns
    ///
    /// `(left_tail, right_tail)` -- the flushed tail audio. May be shorter
    /// than `DEFAULT_FLUSH_SAMPLES` if the pipeline produces fewer samples.
    pub fn flush(&mut self) -> Result<(Vec<f32>, Vec<f32>), KokoroError> {
        self.flush_with_length(DEFAULT_FLUSH_SAMPLES)
    }

    /// Flush with a custom tail length in samples.
    ///
    /// See [`flush`](Self::flush) for details.
    pub fn flush_with_length(
        &mut self,
        tail_samples: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), KokoroError> {
        if tail_samples == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let silence: Vec<Vec<f32>> = vec![vec![0.0f32; tail_samples]; self.n_voices];
        self.pipeline.process(&silence)
    }

    /// Reset all processor state for a new utterance.
    ///
    /// Clears all filter memories, delay lines, envelope followers, etc.
    /// Call this between unrelated audio segments to prevent state leakage
    /// (e.g. reverb tail from utterance A bleeding into utterance B).
    pub fn reset(&mut self) {
        self.pipeline.reset();
        self.total_samples_processed = 0;
    }

    /// Total mono samples processed since creation or last reset.
    #[must_use]
    pub fn total_samples_processed(&self) -> u64 {
        self.total_samples_processed
    }

    /// Total seconds of audio processed since creation or last reset.
    #[must_use]
    pub fn total_seconds_processed(&self) -> f64 {
        self.total_samples_processed as f64 / f64::from(self.sample_rate)
    }

    /// Number of voices this session was configured for.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.n_voices
    }

    /// Sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Reference to the underlying pipeline config.
    #[must_use]
    pub fn config(&self) -> &ChorusMasterConfig {
        self.pipeline.config()
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_streaming_tests.rs"]
mod tests;
