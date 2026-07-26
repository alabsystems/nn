// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Batch inference methods for [`SileroVad`]: full-file probability
//! computation and speech segment detection.
//!
//! Extracted from `silero_vad_forward.rs` to keep files under 400 lines.

use super::segments::{self, SegmentConfig, SpeechSegment};
use super::state::{SileroVadState, CHUNK_SIZE};
use super::SileroVadError;

impl super::SileroVad {
    /// Compute per-chunk speech probabilities for a full audio signal.
    ///
    /// Processes `audio` in 512-sample chunks (32ms at 16kHz), returning one
    /// speech probability per chunk. Trailing samples that don't fill a
    /// complete chunk are zero-padded and processed as a final chunk.
    ///
    /// Resets streaming state before processing (creates a fresh
    /// `SileroVadState::zero()`), so this is suitable for full-file analysis.
    pub fn get_probabilities(
        &self,
        cache: &crate::PipelineCache,
        audio: &[f32],
    ) -> Result<Vec<f32>, SileroVadError> {
        let mut state = SileroVadState::zero();
        let num_full_chunks = audio.len() / CHUNK_SIZE;
        let remainder = audio.len() % CHUNK_SIZE;
        let total_chunks = num_full_chunks + usize::from(remainder > 0);
        let mut probs = Vec::with_capacity(total_chunks);

        for i in 0..num_full_chunks {
            let start = i * CHUNK_SIZE;
            let prob = self.process(cache, &audio[start..start + CHUNK_SIZE], &mut state)?;
            probs.push(prob);
        }

        // Process trailing samples by zero-padding to a full chunk.
        if remainder > 0 {
            let mut padded = vec![0.0f32; CHUNK_SIZE];
            padded[..remainder].copy_from_slice(&audio[num_full_chunks * CHUNK_SIZE..]);
            let prob = self.process(cache, &padded, &mut state)?;
            probs.push(prob);
        }

        Ok(probs)
    }

    /// Detect speech segments in a full audio signal.
    ///
    /// Runs `get_probabilities` then applies the segment detection state
    /// machine with the given `config`. Returns segments with sample-level
    /// boundaries suitable for slicing the original audio.
    ///
    /// `sample_rate` is the audio sample rate in Hz (typically 16000).
    pub fn get_speech_segments(
        &self,
        cache: &crate::PipelineCache,
        audio: &[f32],
        config: &SegmentConfig,
        sample_rate: u32,
    ) -> Result<Vec<SpeechSegment>, SileroVadError> {
        let probs = self.get_probabilities(cache, audio)?;
        Ok(segments::segments_from_probs(
            &probs,
            audio.len(),
            config,
            CHUNK_SIZE,
            sample_rate,
        ))
    }
}
