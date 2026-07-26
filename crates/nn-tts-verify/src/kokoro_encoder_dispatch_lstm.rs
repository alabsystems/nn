// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reusable LSTM dispatch builders for Kokoro encoder.
//!
//! Shared by `kokoro_encoder_dispatch_builders.rs` (TextEncoder BiLSTM)
//! and `kokoro_encoder_dispatch_builders_prosody.rs` (ProsodyPredictor LSTM,
//! F0EnergyPredictor BiLSTM).

use crate::dispatch_builder::DispatchBuilder;

// ---------------------------------------------------------------------------
// LSTM decomposition (reusable across stages)
// ---------------------------------------------------------------------------

/// LSTM cell as 12 primitive dispatch steps.
///
/// Linear(ih) + Linear(hh) + BinaryAdd + 3×Sigmoid + Tanh
/// + 2×BinaryMul + BinaryAdd + Tanh + BinaryMul = 12 steps.
pub(super) fn build_lstm_cell(
    b: &mut DispatchBuilder,
    prefix: &str,
    input_size: usize,
    hidden_size: usize,
) {
    let four_h = 4 * hidden_size;

    b.linear(format!("{prefix}_ih"), input_size, four_h, 1);
    b.linear(format!("{prefix}_hh"), hidden_size, four_h, 1);
    b.binary_add(format!("{prefix}_gate_add"), four_h);
    b.sigmoid(format!("{prefix}_i_gate"), hidden_size);
    b.sigmoid(format!("{prefix}_f_gate"), hidden_size);
    b.tanh(format!("{prefix}_g_gate"), hidden_size);
    b.sigmoid(format!("{prefix}_o_gate"), hidden_size);
    b.binary_mul(format!("{prefix}_f_c"), hidden_size);
    b.binary_mul(format!("{prefix}_i_g"), hidden_size);
    b.binary_add(format!("{prefix}_c_new"), hidden_size);
    b.tanh(format!("{prefix}_tanh_c"), hidden_size);
    b.binary_mul(format!("{prefix}_h_new"), hidden_size);
}

/// BiLSTM = forward LSTM + backward LSTM = 24 steps.
pub(super) fn build_bilstm(
    b: &mut DispatchBuilder,
    prefix: &str,
    input_size: usize,
    hidden_size: usize,
) {
    build_lstm_cell(b, &format!("{prefix}_fwd"), input_size, hidden_size);
    build_lstm_cell(b, &format!("{prefix}_bwd"), input_size, hidden_size);
}
