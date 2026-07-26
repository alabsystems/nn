// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Prosody and F0/Energy predictor builder functions for Kokoro encoder dispatch.
//!
//! Extracted from `kokoro_encoder_dispatch_builders.rs` for the 500-line limit.
//! Covers ProsodyPredictor (3 blocks + dur_proj) and F0EnergyPredictor
//! (shared BiLSTM + 2 parallel AdainResBlk1d heads).

use crate::dispatch_builder::DispatchBuilder;

use super::lstm::{build_bilstm, build_lstm_cell};
use super::{D_EN, F0_HIDDEN, N_PROSODY_LAYERS, PROSODY_LSTM_HIDDEN, STYLE_DIM};

// ---------------------------------------------------------------------------
// Stage 4: ProsodyPredictor
// ---------------------------------------------------------------------------

/// One ProsodyBlock:
/// Conv1d(512,512,k=3,pad=1) + AdaLayerNorm(style_proj Linear) + LN(Sigmoid)
/// + LSTM(input=640, hidden=256, 12 steps) + Linear(512→256) + BinaryAdd(residual)
///   = 1 + 1 + 1 + 12 + 1 + 1 = 17 steps.
pub(super) fn build_prosody_block(b: &mut DispatchBuilder, block_idx: usize) {
    let prefix = format!("prosody_b{block_idx}");

    b.conv1d(format!("{prefix}_conv"), D_EN, D_EN, 3, 1, 1, 1, 1);
    b.linear(format!("{prefix}_ada_style"), STYLE_DIM, 2 * D_EN, 1);
    b.sigmoid(format!("{prefix}_ada_ln"), D_EN);

    // LSTM(input=D_EN+STYLE_DIM=640, hidden=256)
    build_lstm_cell(
        b,
        &format!("{prefix}_lstm"),
        D_EN + STYLE_DIM,
        PROSODY_LSTM_HIDDEN,
    );

    b.linear(
        format!("{prefix}_lstm_proj"),
        2 * PROSODY_LSTM_HIDDEN,
        PROSODY_LSTM_HIDDEN,
        1,
    );
    b.binary_add(format!("{prefix}_residual"), D_EN);
}

/// ProsodyPredictor: N_PROSODY_LAYERS × ProsodyBlock + dur_proj Linear(512→1).
///
/// Steps: 3 × 17 + 1 = 52.
pub(super) fn build_prosody_predictor(b: &mut DispatchBuilder) {
    for i in 0..N_PROSODY_LAYERS {
        build_prosody_block(b, i);
    }
    b.linear("prosody_dur_proj", D_EN, 1, 1);
}

// ---------------------------------------------------------------------------
// Stage 5: F0EnergyPredictor
// ---------------------------------------------------------------------------

/// One AdainResBlk1d block:
/// AdaIN1(Linear) + LeakyReLU(Sigmoid) + Conv1d + AdaIN2(Linear)
/// + LeakyReLU(Sigmoid) + Conv1d + [optional skip Conv1d] + [optional upsample] + BinaryAdd
///   = 7, 8, or 9 steps.
fn build_adain_resblk(
    b: &mut DispatchBuilder,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    has_upsample: bool,
) {
    b.linear(format!("{prefix}_adain1"), STYLE_DIM, 2 * in_ch, 1);
    b.sigmoid(format!("{prefix}_lrelu1"), in_ch);
    b.conv1d(format!("{prefix}_conv1"), in_ch, out_ch, 3, 1, 1, 1, 1);
    b.linear(format!("{prefix}_adain2"), STYLE_DIM, 2 * out_ch, 1);
    b.sigmoid(format!("{prefix}_lrelu2"), out_ch);
    b.conv1d(format!("{prefix}_conv2"), out_ch, out_ch, 3, 1, 1, 1, 1);

    if in_ch != out_ch {
        b.conv1d(format!("{prefix}_skip"), in_ch, out_ch, 1, 1, 1, 0, 1);
    }

    if has_upsample {
        b.conv_transpose1d(format!("{prefix}_upsample"), out_ch, out_ch, 3, 1, 2, 1);
    }

    b.binary_add(format!("{prefix}_residual"), out_ch);
}

/// F0/Energy head: 3 AdainResBlk1d blocks + Conv1d(k=1) projection.
///
/// Block 0: 512→512 (same-ch, no upsample) = 7 steps
/// Block 1: 512→256 (skip_conv + upsample) = 7 + 1(skip) + 1(upsample) = 9 steps
/// Block 2: 256→256 (same-ch, no upsample) = 7 steps
/// + proj Conv1d(k=1) = 1 step  (#3512: replaces Linear, eliminates 4 transpose dispatches)
///   Total per head = 24 steps.
fn build_f0_energy_head(b: &mut DispatchBuilder, head_name: &str) {
    build_adain_resblk(b, &format!("{head_name}_b0"), D_EN, D_EN, false);
    build_adain_resblk(b, &format!("{head_name}_b1"), D_EN, F0_HIDDEN, true);
    build_adain_resblk(b, &format!("{head_name}_b2"), F0_HIDDEN, F0_HIDDEN, false);
    // Conv1d(k=1) projection: equivalent to Linear but operates on [B, C, T] directly,
    // avoiding transpose→Linear→transpose. (#3512)
    b.conv1d(format!("{head_name}_proj"), F0_HIDDEN, 1, 1, 1, 1, 0, 1);
}

/// F0EnergyPredictor: shared BiLSTM(24 steps) + F0 head(24 steps)
/// + Energy head(24 steps) = 72 steps.
pub(super) fn build_f0_energy_predictor(b: &mut DispatchBuilder) {
    build_bilstm(b, "f0_energy_bilstm", D_EN, F0_HIDDEN);
    build_f0_energy_head(b, "f0");
    build_f0_energy_head(b, "energy");
}
