// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming state and output types for Silero VAD.
//!
//! Extracted from `silero_vad.rs` to stay within the 500-line file limit.

/// Hidden size for Silero VAD 16kHz LSTM.
pub(crate) const LSTM_HIDDEN_SIZE: usize = 128;

/// Number of audio context samples carried between chunks.
pub(crate) const AUDIO_CONTEXT_SIZE: usize = 64;

/// Number of new audio samples per chunk (512 for 16kHz, 32ms).
pub(crate) const CHUNK_SIZE: usize = 512;

/// LSTM hidden/cell state and audio context for streaming Silero VAD inference.
///
/// Silero VAD is a streaming model: LSTM h_state, c_state, and audio context
/// persist across 32ms audio chunks. Pass `SileroVadState::zero()` for the
/// first chunk, then feed the returned state into subsequent calls.
///
/// The `context` field holds the last 64 audio samples from the previous chunk,
/// needed by the STFT for windowing continuity across chunk boundaries.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub struct SileroVadState {
    /// LSTM hidden state `[128]`.
    pub h_state: Vec<f32>,
    /// LSTM cell state `[128]`.
    pub c_state: Vec<f32>,
    /// Audio context: last 64 samples from the previous chunk.
    /// Prepended to the next chunk's 512 new samples to form 576 STFT input.
    pub context: Vec<f32>,
}

impl SileroVadState {
    /// Create state from serialized LSTM hidden/cell state and audio context.
    ///
    /// Validates vector lengths against the expected model dimensions:
    /// - `h_state`: `LSTM_HIDDEN_SIZE` (128)
    /// - `c_state`: `LSTM_HIDDEN_SIZE` (128)
    /// - `context`: `AUDIO_CONTEXT_SIZE` (64)
    pub fn new(
        h_state: Vec<f32>,
        c_state: Vec<f32>,
        context: Vec<f32>,
    ) -> Result<Self, super::SileroVadError> {
        if h_state.len() != LSTM_HIDDEN_SIZE {
            return Err(super::SileroVadError::StateDimension {
                field: "h_state",
                expected: LSTM_HIDDEN_SIZE,
                actual: h_state.len(),
            });
        }
        if c_state.len() != LSTM_HIDDEN_SIZE {
            return Err(super::SileroVadError::StateDimension {
                field: "c_state",
                expected: LSTM_HIDDEN_SIZE,
                actual: c_state.len(),
            });
        }
        if context.len() != AUDIO_CONTEXT_SIZE {
            return Err(super::SileroVadError::StateDimension {
                field: "context",
                expected: AUDIO_CONTEXT_SIZE,
                actual: context.len(),
            });
        }
        Ok(Self {
            h_state,
            c_state,
            context,
        })
    }

    /// Create zero-initialized state for the first audio chunk.
    pub fn zero() -> Self {
        Self {
            h_state: vec![0.0f32; LSTM_HIDDEN_SIZE],
            c_state: vec![0.0f32; LSTM_HIDDEN_SIZE],
            context: vec![0.0f32; AUDIO_CONTEXT_SIZE],
        }
    }
}

/// Output of a single Silero VAD forward pass.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub struct SileroVadOutput {
    /// Speech probability in `[0.0, 1.0]`.
    pub probability: f32,
    /// Updated LSTM state to pass into the next chunk.
    pub state: SileroVadState,
}

impl SileroVadOutput {
    /// Create a new VAD output.
    pub fn new(probability: f32, state: SileroVadState) -> Self {
        Self { probability, state }
    }
}
