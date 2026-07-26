// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Silero VAD architecture helpers for timing certificate tests.
//!
//! Builds `VerifiedStage` pipeline stages matching the Silero VAD architecture:
//! - **Encoder stage**: STFT magnitude `[1, 129, T]` → Conv1d+ReLU blocks → `[1, 128]`
//! - **LSTM + output stage**: `[1, 128]` → LSTM cell → ReLU + Linear + Sigmoid → `[1, 1]`
//!
//! Part of #1739 Phase 37.

use crate::pipeline::VerifiedStage;

/// Build VerifiedStage pipeline stages matching the Silero VAD architecture.
///
/// Models the VAD as two stages with compatible junction bounds:
/// 1. **Encoder** (STFT mag → 128-dim representation via Conv1d+ReLU + temporal pool)
/// 2. **LSTM + Output** (128-dim → scalar VAD probability via LSTM + Linear + Sigmoid)
///
/// The junction between stages has encoder output ⊆ lstm_output input ([-0.8, 0.8] ⊆ [-1.0, 1.0]).
///
/// # Arguments
///
/// * `encoder_dim` — Dimensionality for bounds vectors (128 for encoder output).
pub(super) fn silero_vad_verified_stages(encoder_dim: usize) -> Vec<VerifiedStage> {
    // STFT input: 129-frequency magnitude bins, values in [0, ~10] for speech.
    // Encoder output: 128-channel representation after temporal pooling, bounded
    // by Conv1d+ReLU (ReLU clamps to [0, ∞), empirically within [-0.8, 0.8]
    // after normalization).
    let stft_input_dim = 129;

    vec![
        VerifiedStage {
            name: "vad_encoder".to_string(),
            input_lower: vec![0.0; stft_input_dim],
            input_upper: vec![10.0; stft_input_dim],
            output_lower: vec![-0.8; encoder_dim],
            output_upper: vec![0.8; encoder_dim],
            input_shape: vec![1, stft_input_dim],
            output_shape: vec![1, encoder_dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "vad_lstm_output".to_string(),
            input_lower: vec![-1.0; encoder_dim],
            input_upper: vec![1.0; encoder_dim],
            output_lower: vec![0.0],
            output_upper: vec![1.0],
            input_shape: vec![1, encoder_dim],
            output_shape: vec![1, 1],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ]
}
