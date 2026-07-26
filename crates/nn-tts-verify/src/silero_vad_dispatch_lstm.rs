// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM cell decomposed dispatch steps for Silero VAD cost model.
//!
//! The LSTM cell is modeled as its decomposed primitives: 2 gate Linears
//! (W_ih + W_hh) + BinaryAdd + 3 Sigmoid + Tanh + 2 BinaryMul + BinaryAdd
//! + Tanh + BinaryMul = 12 steps.
//!
//! Extracted from `silero_vad_dispatch.rs` to keep both files under 500 lines.

use super::LSTM_HIDDEN;
use crate::dispatch_builder::DispatchBuilder;

/// Build the full LSTM cell: gate linears + activations + state update = 12 steps.
pub(super) fn build_lstm_cell(b: &mut DispatchBuilder) {
    let four_h = 4 * LSTM_HIDDEN;

    // Gate linears: W_ih + W_hh + gate add
    b.linear("lstm_ih", LSTM_HIDDEN, four_h, 1);
    b.linear("lstm_hh", LSTM_HIDDEN, four_h, 1);
    b.binary_add("lstm_gate_add", four_h);

    // Gate activations
    b.sigmoid("lstm_i_gate", LSTM_HIDDEN);
    b.sigmoid("lstm_f_gate", LSTM_HIDDEN);
    b.tanh("lstm_g_gate", LSTM_HIDDEN);
    b.sigmoid("lstm_o_gate", LSTM_HIDDEN);

    // Cell state update: c_new = f*c_old + i*g
    b.binary_mul("lstm_f_c", LSTM_HIDDEN);
    b.binary_mul("lstm_i_g", LSTM_HIDDEN);
    b.binary_add("lstm_c_new", LSTM_HIDDEN);

    // Hidden state update: h_new = o * tanh(c_new)
    b.tanh("lstm_tanh_c", LSTM_HIDDEN);
    b.binary_mul("lstm_h_new", LSTM_HIDDEN);
}
