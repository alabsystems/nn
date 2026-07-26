// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro TTS pre-vocoder encoder dispatch plan for cost model profiling.
//!
//! Converts the 5 Kokoro encoder stages into a `Vec<DispatchStep>` suitable
//! for roofline profiling, timing certification, and calibration against
//! measured GPU times.
//!
//! ## Stages
//!
//! 1. **PlBert (ALBERT)**: Factorized embeddings (3 Embedding + BinaryAdd×2
//!    + LayerNorm + Linear(128→768)) + 12 shared transformer layers
//!      (Q/K/V/dense Linear + MatMul×2 + Softmax + BinaryAdd + LN + FFN + LN).
//! 2. **bert_encoder**: Linear(768→512) projection from ALBERT to encoder dim.
//! 3. **TextEncoder**: BiLSTM(512→256×2) + Linear(512→512) projection.
//! 4. **ProsodyPredictor**: 3 ProsodyBlocks (Conv1d + AdaLayerNorm + LSTM +
//!    Linear + residual) + duration projection Linear(512→1).
//! 5. **F0EnergyPredictor**: Shared BiLSTM(512→256×2) + F0 head
//!    (3 AdainResBlk1d + Linear) + Energy head (3 AdainResBlk1d + Linear).
//!
//! Part of #1739 AC3 and #1741 P5.

use nn_dsl::DispatchStep;

use crate::dispatch_builder::DispatchBuilder;

#[path = "kokoro_encoder_dispatch_lstm.rs"]
mod lstm;

#[path = "kokoro_encoder_dispatch_builders.rs"]
mod builders;

#[path = "kokoro_encoder_dispatch_builders_prosody.rs"]
mod prosody;

use builders::{
    build_bert_encoder, build_plbert_embeddings, build_plbert_layer, build_text_encoder,
};
use prosody::{build_f0_energy_predictor, build_prosody_predictor};

// ---------------------------------------------------------------------------
// Architecture constants — PlBert (ALBERT)
// ---------------------------------------------------------------------------

/// ALBERT factorized embedding dimension.
pub(super) const ALBERT_EMB_DIM: usize = 128;
/// ALBERT hidden dimension.
pub(super) const ALBERT_HIDDEN: usize = 768;
/// ALBERT number of attention heads.
pub(super) const ALBERT_NUM_HEADS: usize = 12;
/// ALBERT FFN intermediate dimension.
pub(super) const ALBERT_FFN_DIM: usize = 2048;
/// Number of shared ALBERT layers.
const ALBERT_LAYERS: usize = 12;

// ---------------------------------------------------------------------------
// Architecture constants — Encoder / Prosody / F0
// ---------------------------------------------------------------------------

/// Encoder dimension (bert_encoder output, TextEncoder BiLSTM input).
pub(super) const D_EN: usize = 512;
/// Style embedding dimension.
pub(super) const STYLE_DIM: usize = 128;
/// Number of ProsodyPredictor blocks.
pub(super) const N_PROSODY_LAYERS: usize = 3;
/// ProsodyPredictor LSTM hidden size per direction.
pub(super) const PROSODY_LSTM_HIDDEN: usize = 256;
/// F0/Energy predictor hidden dimension (AdainResBlk1d channel after downscale).
pub(super) const F0_HIDDEN: usize = 256;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the full Kokoro pre-vocoder encoder dispatch plan for `text_tokens`
/// input phoneme tokens.
///
/// Stages (in order):
/// 1. PlBert embeddings (7 steps)
/// 2. PlBert shared transformer × 12 (12 × 14 = 168 steps)
/// 3. bert_encoder Linear(768→512) (1 step)
/// 4. TextEncoder BiLSTM + Linear (25 steps)
/// 5. ProsodyPredictor: 3 blocks + dur_proj (52 steps)
/// 6. F0EnergyPredictor: BiLSTM + F0 head + Energy head (72 steps)
///
/// Returns `(dispatch_plan, node_count)`.
pub fn build_kokoro_encoder_dispatch_plan(text_tokens: usize) -> (Vec<DispatchStep>, usize) {
    let mut b = DispatchBuilder::with_capacity(TOTAL_EXPECTED_STEPS + 16);

    build_plbert_embeddings(&mut b, text_tokens);
    for layer_idx in 0..ALBERT_LAYERS {
        build_plbert_layer(&mut b, layer_idx, text_tokens);
    }
    build_bert_encoder(&mut b, text_tokens);
    build_text_encoder(&mut b);
    build_prosody_predictor(&mut b);
    build_f0_energy_predictor(&mut b);

    let node_count = b.node_count();
    (b.into_steps(), node_count)
}

/// Build the standard Kokoro encoder dispatch plan with 100 input tokens.
pub fn build_kokoro_encoder_dispatch_plan_default() -> (Vec<DispatchStep>, usize) {
    build_kokoro_encoder_dispatch_plan(100)
}

// ---------------------------------------------------------------------------
// Step count constants (validated by tests)
// ---------------------------------------------------------------------------

/// PlBert embedding layer: 3 Embedding + 2 BinaryAdd + 1 LayerNorm + 1 Linear = 7.
pub const PLBERT_EMB_STEPS: usize = 7;
/// One ALBERT layer: 6 Linear(Q+K+V+Dense+FFN_up+FFN_down) + 2 MatMul
/// + 1 Softmax + 2 BinaryAdd + 2 Sigmoid + 1 GELU = 14.
pub const PLBERT_LAYER_STEPS: usize = 14;
/// Number of shared ALBERT layers.
pub const NUM_ALBERT_LAYERS: usize = ALBERT_LAYERS;
/// bert_encoder: Linear(768→512) = 1 step.
pub const BERT_ENCODER_STEPS: usize = 1;
/// TextEncoder: BiLSTM(24) + Linear(1) = 25 steps.
pub const TEXT_ENCODER_STEPS: usize = 25;
/// ProsodyPredictor: 3 blocks × 17 + 1 dur_proj = 52 steps.
pub const PROSODY_PREDICTOR_STEPS: usize = 52;
/// F0EnergyPredictor: BiLSTM(24) + F0 head(24) + Energy head(24) = 72 steps.
pub const F0_ENERGY_PREDICTOR_STEPS: usize = 72;

/// Total expected steps for the full encoder dispatch plan.
///
/// 7 (emb) + 168 (12×14 layers) + 1 (bert_enc) + 25 (text_enc)
/// + 52 (prosody) + 72 (f0_energy) = 325.
pub const TOTAL_EXPECTED_STEPS: usize = PLBERT_EMB_STEPS
    + NUM_ALBERT_LAYERS * PLBERT_LAYER_STEPS
    + BERT_ENCODER_STEPS
    + TEXT_ENCODER_STEPS
    + PROSODY_PREDICTOR_STEPS
    + F0_ENERGY_PREDICTOR_STEPS;

#[cfg(test)]
#[path = "kokoro_encoder_dispatch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kokoro_encoder_dispatch_tests_profiling.rs"]
mod tests_profiling;
