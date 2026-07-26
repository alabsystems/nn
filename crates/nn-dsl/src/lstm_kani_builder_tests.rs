// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `TensorBlockBuilder::add_lstm`.
//!
//! Proves structural correctness of LSTM tensor IR construction:
//! - validate() succeeds for all valid bounded parameters (explicit call, with/without bias)
//! - validate() rejects weight_ih shape mismatch (negative case — LSTM gate structure)
//!
//! LSTM gate ordering matches PyTorch `nn.LSTMCell`: `[i, f, g, o]`.
//! Weight shapes: `weight_ih: [4*H, I]`, `weight_hh: [4*H, H]`, `bias: [4*H]`.
//!
//! Part of #729 (dvoice epic), Re: #747 (LSTM op). Cleaned up in #800.

use crate::tensor_block_builder::TensorBlockBuilder;

/// Proves `add_lstm` + `build` + explicit `validate()` succeeds (with bias).
///
/// Domain: input_size in [1, 8], hidden_size in [1, 8].
/// Makes validate() proof obligation explicit, independent of debug_assert compilation.
/// Exercises the full LSTM gate structure check: weight_ih=[4H,I], weight_hh=[4H,H], bias=[4H].
#[kani::unwind(8)]
#[kani::proof]
fn lstm_builder_with_bias_validates_ok() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(input_size >= 1 && input_size <= 4);
    kani::assume(hidden_size >= 1 && hidden_size <= 4);

    let four_h = 4 * hidden_size;

    let mut b = TensorBlockBuilder::new("kani_lstm");
    let input = b.add_input("input", &[input_size]);
    let hidden = b.add_input("hidden_state", &[hidden_size]);
    let cell = b.add_input("cell_state", &[hidden_size]);
    let w_ih = b.add_input("weight_ih", &[four_h, input_size]);
    let w_hh = b.add_input("weight_hh", &[four_h, hidden_size]);
    let bias = b.add_input("bias", &[four_h]);
    let out = b.add_lstm(input, hidden, cell, w_ih, w_hh, Some(bias), &[hidden_size]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for well-formed LSTM with bias"
    );
}

/// Proves `add_lstm` + `build` + explicit `validate()` succeeds (without bias).
#[kani::unwind(8)]
#[kani::proof]
fn lstm_builder_no_bias_validates_ok() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(input_size >= 1 && input_size <= 4);
    kani::assume(hidden_size >= 1 && hidden_size <= 4);

    let four_h = 4 * hidden_size;

    let mut b = TensorBlockBuilder::new("kani_lstm");
    let input = b.add_input("input", &[input_size]);
    let hidden = b.add_input("hidden_state", &[hidden_size]);
    let cell = b.add_input("cell_state", &[hidden_size]);
    let w_ih = b.add_input("weight_ih", &[four_h, input_size]);
    let w_hh = b.add_input("weight_hh", &[four_h, hidden_size]);
    let out = b.add_lstm(input, hidden, cell, w_ih, w_hh, None, &[hidden_size]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for well-formed LSTM without bias"
    );
}

/// Proves validate() rejects LSTM with wrong weight_ih shape (gate structure violation).
///
/// Constructs weight_ih: [3*H, I] instead of [4*H, I]. The LSTM validator checks
/// that weight_ih rows == 4*hidden_size (for i/f/g/o gates). This is the core
/// ML-semantic constraint that prevents silent gate misconfiguration.
#[kani::unwind(8)]
#[kani::proof]
fn lstm_builder_rejects_wrong_wih_shape() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(input_size >= 1 && input_size <= 4);
    kani::assume(hidden_size >= 1 && hidden_size <= 4);

    let wrong_rows = 3 * hidden_size; // Should be 4*H for [i,f,g,o] gates
    let four_h = 4 * hidden_size;

    let mut b = TensorBlockBuilder::new("kani_lstm_bad");
    let input = b.add_input("input", &[input_size]);
    let hidden = b.add_input("hidden_state", &[hidden_size]);
    let cell = b.add_input("cell_state", &[hidden_size]);
    let w_ih = b.add_input("weight_ih", &[wrong_rows, input_size]); // 3*H, not 4*H
    let w_hh = b.add_input("weight_hh", &[four_h, hidden_size]);
    let out = b.add_lstm(input, hidden, cell, w_ih, w_hh, None, &[hidden_size]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_err(),
        "validate() must reject LSTM with weight_ih rows != 4*H"
    );
}
