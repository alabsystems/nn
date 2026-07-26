// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder functions for Kokoro pre-vocoder encoder stages 1–3.
//!
//! Stage 1: PlBert (ALBERT) — factorized embeddings + 12 shared transformer layers
//! Stage 2: bert_encoder — Linear projection from ALBERT hidden to encoder dim
//! Stage 3: TextEncoder — BiLSTM + projection
//!
//! Extracted from `kokoro_encoder_dispatch.rs` for the 500-line limit.

use crate::dispatch_builder::DispatchBuilder;

use super::lstm::build_bilstm;
use super::{ALBERT_EMB_DIM, ALBERT_FFN_DIM, ALBERT_HIDDEN, ALBERT_NUM_HEADS, D_EN};

// ---------------------------------------------------------------------------
// Stage 1: PlBert (ALBERT)
// ---------------------------------------------------------------------------

/// PlBert embedding layer: 3 Embedding lookups + 2 BinaryAdd + LayerNorm(Sigmoid proxy)
/// + Linear(128→768) = 7 steps.
pub(super) fn build_plbert_embeddings(b: &mut DispatchBuilder, seq_len: usize) {
    b.embedding("plbert_word_emb", ALBERT_EMB_DIM, seq_len);
    b.embedding("plbert_pos_emb", ALBERT_EMB_DIM, seq_len);
    b.embedding("plbert_tt_emb", ALBERT_EMB_DIM, seq_len);
    b.binary_add("plbert_emb_add1", seq_len * ALBERT_EMB_DIM);
    b.binary_add("plbert_emb_add2", seq_len * ALBERT_EMB_DIM);
    b.sigmoid("plbert_emb_ln", seq_len * ALBERT_EMB_DIM);
    b.linear("plbert_emb_proj", ALBERT_EMB_DIM, ALBERT_HIDDEN, seq_len);
}

/// One ALBERT attention layer (shared 12 times):
/// Q/K/V/dense Linear(768→768) + MatMul(Q×K^T) + Softmax + MatMul(attn×V)
/// + BinaryAdd(residual) + LN + FFN_up + GELU + FFN_down + BinaryAdd + LN
///   = 6 Linear + 2 MatMul + 1 Softmax + 2 BinaryAdd + 2 Sigmoid + 1 GELU = 14 steps.
pub(super) fn build_plbert_layer(b: &mut DispatchBuilder, layer_idx: usize, seq_len: usize) {
    let d = ALBERT_HIDDEN;
    let ffn = ALBERT_FFN_DIM;
    let prefix = format!("plbert_l{layer_idx}");
    let head_dim = d / ALBERT_NUM_HEADS;

    // Q/K/V projections
    b.linear(format!("{prefix}_q"), d, d, seq_len);
    b.linear(format!("{prefix}_k"), d, d, seq_len);
    b.linear(format!("{prefix}_v"), d, d, seq_len);

    // MatMul(Q, K^T) → attention scores
    b.matmul(
        format!("{prefix}_qk"),
        seq_len,
        head_dim,
        seq_len,
        ALBERT_NUM_HEADS,
        true,
        false,
        None,
    );

    // Softmax + MatMul(attn, V)
    b.softmax(
        format!("{prefix}_softmax"),
        seq_len,
        ALBERT_NUM_HEADS * seq_len,
    );
    b.matmul(
        format!("{prefix}_av"),
        seq_len,
        seq_len,
        head_dim,
        ALBERT_NUM_HEADS,
        false,
        false,
        None,
    );

    // Dense + residual + LN
    b.linear(format!("{prefix}_dense"), d, d, seq_len);
    b.binary_add(format!("{prefix}_attn_res"), seq_len * d);
    b.sigmoid(format!("{prefix}_attn_ln"), seq_len * d);

    // FFN: up + GELU + down + residual + LN
    b.linear(format!("{prefix}_ffn_up"), d, ffn, seq_len);
    b.gelu(format!("{prefix}_gelu"), seq_len * ffn);
    b.linear(format!("{prefix}_ffn_down"), ffn, d, seq_len);
    b.binary_add(format!("{prefix}_ffn_res"), seq_len * d);
    b.sigmoid(format!("{prefix}_ffn_ln"), seq_len * d);
}

// ---------------------------------------------------------------------------
// Stage 2: bert_encoder — Linear(768→512)
// ---------------------------------------------------------------------------

/// bert_encoder: Linear(768→512) = 1 step.
pub(super) fn build_bert_encoder(b: &mut DispatchBuilder, seq_len: usize) {
    b.linear("bert_encoder", ALBERT_HIDDEN, D_EN, seq_len);
}

// ---------------------------------------------------------------------------
// Stage 3: TextEncoder — BiLSTM(512→256×2) + Linear(512→512)
// ---------------------------------------------------------------------------

/// TextEncoder: BiLSTM(24 steps) + Linear(512→512) = 25 steps.
pub(super) fn build_text_encoder(b: &mut DispatchBuilder) {
    build_bilstm(b, "text_enc", D_EN, D_EN / 2);
    b.linear("text_enc_proj", D_EN, D_EN, 1);
}
